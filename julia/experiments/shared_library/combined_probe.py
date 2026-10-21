"""Exercise one combined constructor/HiGHS image through both Python interfaces.

This is an experimental adapter, not a new Model.constraint(func, set) API.
Every mode runs in its own process; ``all`` includes both startup orders and
safe rejection of JuliaCall already initialized with its default image.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from probe import (
    address,
    bind_constructors,
    bind_runtime,
    juliacall_runtime,
    julia_type_tokens,
    library_type_tokens,
    load_library,
    report,
    require,
)


MODES = ("standalone", "shared", "compiled-first", "reject-existing")


def guard_shared_image(jl, runtime, lib):
    """Reject the wrong active image before invoking any compiled Julia export."""
    require(
        address(lib.jl_is_initialized) == address(runtime.jl_is_initialized),
        "The compiled library and JuliaCall resolved different Julia runtimes",
    )
    require(
        bool(jl.seval(
            "isdefined(Main, :JuMPyHiGHS) && "
            "isdefined(Main, :JuMPyMOIProbe) && "
            "isdefined(Main, :JuMPyCombinedProbe)"
        )),
        "JuliaCall is already initialized with a different system image; "
        "restart with PYTHON_JULIACALL_SYSIMAGE pointing to the combined library",
    )
    active_image = str(jl.seval(
        'let path = Base.JLOptions().image_file; '
        'path == C_NULL ? "" : unsafe_string(path); end'
    ))
    require(
        bool(active_image) and Path(active_image).resolve() == Path(lib._name).resolve(),
        "JuliaCall is already initialized with a different system image artifact; "
        "refusing exports from an image that the active runtime did not load",
    )


def backend_library(path, *, fresh):
    """Reuse JuMPy's ABI bindings, but keep the GIL for shared PythonCall calls."""
    from jumpy.backend import _init_lib

    lib = load_library(path)
    bind_runtime(lib)
    if fresh:
        require(lib.jl_is_initialized() == 0, "Expected a fresh Julia runtime")
    # _init_lib's initialization is a no-op when JuliaCall already loaded this
    # image. Its callers must have checked guard_shared_image first in that case.
    bound = _init_lib(str(path))
    for name, function in vars(bound).items():
        if name.startswith("jumpy_"):
            target = getattr(lib, name)
            target.argtypes, target.restype = function.argtypes, function.restype
    bind_constructors(lib)
    lib.jumpy_probe_add_constraint.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
    ]
    lib.jumpy_probe_add_constraint.restype = ctypes.c_longlong
    return lib


def new_set(lib, bound=1.0):
    handle = lib.jumpy_moi_new_less_than(bound)
    require(handle != 0, "Compiled LessThan constructor failed")
    require(lib.jumpy_moi_kind(handle) == 1, "Constructor returned the wrong set type")
    lib.jumpy_moi_collect()
    require(lib.jumpy_moi_bound(handle) == bound, "Constructed set did not survive GC")
    return handle


def reject_invalid_handles(lib, model, expression, handle):
    """Exercise checked errors without dereferencing foreign or stale pointers."""
    from jumpy.backend import JuliacOps

    add = lib.jumpy_probe_add_constraint
    require(add(None, expression.moi, handle) == -1, "Null model was accepted")
    require(add(model._ops._m, None, handle) == -1, "Null function was accepted")
    require(add(model._ops._m, expression.moi, 0) == -1, "Invalid set handle was accepted")
    nonset = lib.jumpy_moi_new_constant(1.0)
    require(nonset != 0, "Constant constructor failed")
    try:
        require(add(model._ops._m, expression.moi, nonset) == -1,
                "A constant was accepted as a set")
    finally:
        require(lib.jumpy_moi_release(nonset) == 0, "Could not release the constant")
    require(add(model._ops._m, expression.moi, nonset) == -1,
            "A released set handle was accepted")
    other = JuliacOps(lib)
    try:
        require(add(other._m, expression.moi, handle) == -1,
                "A function belonging to a different model was accepted")
    finally:
        other.free()


def solve_compiled(lib, handle, *, bound=1.0, jl=None, native_set=None):
    import jumpy as jp
    from jumpy.backend import JuliacOps

    model = jp.Model(backend=JuliacOps(lib))
    try:
        x, y = model.variables(2, lower=0)
        expression = x + y
        reject_invalid_handles(lib, model, expression, handle)
        # Experimental set-handle adapter: the export consumes the actual MOI
        # set, instead of reconstructing it from a sense code and RHS.
        index = lib.jumpy_probe_add_constraint(
            model._ops._m, expression.moi, handle,
        )
        require(index >= 0, "Compiled backend rejected the constructor's MOI set")
        if jl is not None:
            same_set = jl.seval(
                "(s, h) -> Main.JuMPyCombinedProbe.LAST_SET[] === s && "
                "s === Main.JuMPyMOIProbe.get_object(UInt64(h))"
            )
            # LessThan{Float64} is immutable: === checks native type/value,
            # not physical allocation identity. The export itself performs no
            # serialization or set reconstruction.
            require(bool(same_set(native_set, handle)),
                    "Compiled consumer received a different native set")
        model.objective = jp.maximize(expression)
        model.optimize()
        optimum = model.value(x) + model.value(y)
        require(math.isclose(optimum, bound), f"Wrong compiled optimum: {optimum}")
        return optimum
    finally:
        model.close()


def prepare_juliacall_ops(jl):
    # The production JuliaCall loader includes source for independent use.
    # This experiment instead deliberately uses the module baked into the image.
    import jumpy.bridge_juliacall as bridge

    jl.seval("import GenOpt; import HiGHS")
    jl.seval(
        "if !isdefined(Main, :JuMPyMOI); "
        "const JuMPyMOI = JuMPyHiGHS.JuMPyMOI; end"
    )
    require(bool(jl.seval(
        "Main.JuMPyMOI === Main.JuMPyHiGHS.JuMPyMOI === "
        "Main.JuMPyMOIProbe.JuMPyMOI"
    )), "The interfaces are not using the same constructor module")
    bridge._JL = jl


def solve_juliacall(jl, native_set, *, bound=1.0):
    import jumpy as jp
    from jumpy.bridge_juliacall import JuliaCallOps

    model = jp.Model(backend=JuliaCallOps())
    try:
        x, y = model.variables(2, lower=0)
        expression = x + y
        # The public two-argument Model.constraint interface does not exist yet.
        # Pass the genuine Julia value directly through MOI for this experiment.
        jl.MOI.Utilities.normalize_and_add_constraint(
            model._ops._optimizer,
            model._ops._simplify(expression.moi),
            native_set,
        )
        model.objective = jp.maximize(expression)
        model.optimize()
        optimum = model.value(x) + model.value(y)
        require(math.isclose(optimum, bound), f"Wrong JuliaCall optimum: {optimum}")
        return optimum
    finally:
        model.close()


def standalone(path):
    require("juliacall" not in sys.modules, "JuliaCall was imported before the probe")
    lib = backend_library(path, fresh=True)
    handle = new_set(lib)
    try:
        optimum = solve_compiled(lib, handle)
    finally:
        require(lib.jumpy_moi_release(handle) == 0, "Could not release the set")
    require("juliacall" not in sys.modules, "Compiled backend unexpectedly imported JuliaCall")
    report(
        "standalone", status="passed", optimum=optimum,
        juliacall_imported=False, native_constructor_set_consumed=True,
        invalid_handles_rejected=True,
    )


def shared(path, *, compiled_first=False):
    if compiled_first:
        require("juliacall" not in sys.modules, "JuliaCall imported before the constructor")
        lib = backend_library(path, fresh=True)
        handle = new_set(lib)
        # This is the important opposite startup order: create the set before
        # importing JuliaCall, without changing runtimes or serializing the set.
        jl, runtime = juliacall_runtime()
        guard_shared_image(jl, runtime, lib)
    else:
        jl, runtime = juliacall_runtime()
        guard_shared_image(jl, runtime, load_library(path))
        lib = backend_library(path, fresh=False)
        handle = new_set(lib)
    require(
        library_type_tokens(lib) == julia_type_tokens(jl),
        "Compiled constructors and JuliaCall disagree on native MOI type identity",
    )
    prepare_juliacall_ops(jl)
    borrow = jl.seval("h -> Main.JuMPyMOIProbe.get_object(UInt64(h))")
    retained_set = borrow(handle)
    try:
        compiled_optimum = solve_compiled(lib, handle, jl=jl, native_set=retained_set)
        juliacall_optimum = solve_juliacall(jl, retained_set)
    finally:
        require(lib.jumpy_moi_release(handle) == 0, "Could not release the set handle")
    # Clear the consumer's diagnostic root too, so the PythonCall wrapper is
    # responsible for retaining the borrowed value after its handle is gone.
    jl.seval("Main.JuMPyCombinedProbe.LAST_SET[] = nothing")
    lib.jumpy_moi_collect()
    jl.GC.gc()
    require(retained_set.upper == 1.0, "JuliaCall failed to retain the native set")
    retained_optimum = solve_juliacall(jl, retained_set)

    # The reverse direction also uses the exact Julia object, not a serialized
    # value: JuliaCall constructs it, the common registry roots it, C consumes it.
    reverse_set = jl.MOI.LessThan(2.0)
    keep = jl.seval("s -> Main.JuMPyMOIProbe.keep(s)")
    reverse_handle = int(keep(reverse_set))
    try:
        reverse_optimum = solve_compiled(
            lib, reverse_handle, bound=2.0, jl=jl, native_set=reverse_set,
        )
    finally:
        require(lib.jumpy_moi_release(reverse_handle) == 0, "Could not release reverse set")
        jl.seval("Main.JuMPyCombinedProbe.LAST_SET[] = nothing")
    report(
        "compiled-first" if compiled_first else "shared",
        status="passed", images=1, runtimes=1, same_moi_types=True,
        same_constructor_module=True, same_native_set_in_both_backends=True,
        compiled_optimum=compiled_optimum, juliacall_optimum=juliacall_optimum,
        optimum_after_handle_release=retained_optimum,
        juliacall_created_set_compiled_optimum=reverse_optimum,
        constructor_called_before_juliacall_import=compiled_first,
    )


def reject_existing(path):
    jl, runtime = juliacall_runtime()
    require(not bool(jl.seval("isdefined(Main, :JuMPyCombinedProbe)")),
            "Rejection test needs JuliaCall's default system image")
    lib = load_library(path)
    try:
        guard_shared_image(jl, runtime, lib)
    except RuntimeError as error:
        require("different system image" in str(error), "Unexpected rejection reason")
        report(
            "reject-existing", status="passed", diagnostic=str(error),
            compiled_exports_called=False, live_objects_transferred=False,
        )
    else:
        raise RuntimeError("A different active system image was not rejected")


def run_child(mode, library, timeout):
    env = os.environ.copy()
    env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
    env["PYTHON_JULIACALL_THREADS"] = "1"
    env["PYTHON_JULIACALL_INIT"] = "yes"
    if mode in ("shared", "compiled-first"):
        env["PYTHON_JULIACALL_SYSIMAGE"] = str(library)
    else:
        env.pop("PYTHON_JULIACALL_SYSIMAGE", None)
    if mode == "compiled-first":
        # jl_init_with_image_handle reads JULIA_PROJECT before JuliaCall gets
        # a chance to parse its own options. Select the same environment now,
        # so PythonCall can be loaded later without creating a second runtime.
        project = (
            env.get("PYTHON_JULIACALL_PROJECT")
            or env.get("PYTHON_JULIAPKG_PROJECT")
            or env.get("JULIA_PROJECT")
        )
        if not project:
            report(mode, status="failed", reason=(
                "Set PYTHON_JULIAPKG_PROJECT (or PYTHON_JULIACALL_PROJECT) "
                "to an environment containing PythonCall before this probe"
            ))
            return False
        env["JULIA_PROJECT"] = project
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), mode, str(library), "--worker"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        output, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        output, _ = child.communicate()
        print(output, end="", flush=True)
        report(mode, status="failed", reason=f"Timed out after {timeout} seconds")
        return False
    print(output, end="", flush=True)
    if child.returncode:
        report(mode, status="failed", returncode=child.returncode)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "all"))
    parser.add_argument("library", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    if args.worker:
        require(args.mode != "all", "Workers run exactly one mode")
        if args.mode == "compiled-first":
            shared(library, compiled_first=True)
        else:
            globals()[args.mode.replace("-", "_")](library)
        return 0
    modes = MODES if args.mode == "all" else (args.mode,)
    results = [run_child(mode, library, args.timeout) for mode in modes]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
