"""Probe constructor-library composition in disposable Python processes.

Examples (configure the matching Julia/PythonCall environment first)::

    python probe.py all /absolute/path/libjumpy_moi.so
    python probe.py private /absolute/path/private/lib/libjumpy_moi.so

The shared-sysimage case uses one Julia image. The second-image case does
not call Julia exports from an image that the active runtime did not load.
The private case exchanges primitive values only, never live Julia objects.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


MODES = (
    "standalone",
    "shared-sysimage",
    "second-image",
    "second-compiled-image",
    "private",
)
KINDS = {"set": 1, "variable": 2, "constant": 3, "expression": 4}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def report(mode, **result):
    print(json.dumps({"mode": mode, **result}, sort_keys=True), flush=True)


def load_library(path):
    # Keep the GIL: in shared-sysimage mode this runtime also hosts PythonCall.
    return ctypes.PyDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def bind_runtime(lib):
    lib.jl_is_initialized.argtypes = []
    lib.jl_is_initialized.restype = ctypes.c_int
    lib.jl_init_with_image_handle.argtypes = [ctypes.c_void_p]
    lib.jl_init_with_image_handle.restype = None


def initialize(lib):
    bind_runtime(lib)
    lib.jl_init_with_image_handle(lib._handle)
    require(lib.jl_is_initialized() == 1, "Julia runtime did not initialize")


def address(function):
    # These addresses are compared as integers, never dereferenced.
    return ctypes.cast(function, ctypes.c_void_p).value


def bind_constructors(lib):
    u64, i64 = ctypes.c_uint64, ctypes.c_longlong
    signatures = {
        "jumpy_moi_new_less_than": ([ctypes.c_double], u64),
        "jumpy_moi_new_variable": ([i64], u64),
        "jumpy_moi_new_constant": ([ctypes.c_double], u64),
        "jumpy_moi_new_add": ([u64, u64], u64),
        "jumpy_moi_kind": ([u64], ctypes.c_int),
        "jumpy_moi_bound": ([u64], ctypes.c_double),
        "jumpy_moi_arity": ([u64], i64),
        "jumpy_moi_release": ([u64], ctypes.c_int),
        "jumpy_moi_collect": ([], None),
        "jumpy_moi_type_token": ([ctypes.c_int], ctypes.c_void_p),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result


def make_nodes(lib):
    require(lib.jumpy_moi_new_variable(0) == 0, "Invalid variable index accepted")
    require(lib.jumpy_moi_new_add(0, 0) == 0, "Invalid argument handles accepted")
    require(lib.jumpy_moi_kind(0) == -1, "Invalid handle has a valid kind")
    require(math.isnan(lib.jumpy_moi_bound(0)), "Invalid handle has a valid bound")
    require(lib.jumpy_moi_arity(0) == -1, "Invalid handle has a valid arity")
    nodes = {
        "set": lib.jumpy_moi_new_less_than(4.0),
        "variable": lib.jumpy_moi_new_variable(1),
        "constant": lib.jumpy_moi_new_constant(1.0),
    }
    nodes["expression"] = lib.jumpy_moi_new_add(
        nodes["variable"], nodes["constant"]
    )
    require(all(nodes.values()), "A constructor returned an invalid handle")
    require(len(set(nodes.values())) == 4, "Constructor handles are not distinct")
    for name, handle in nodes.items():
        require(lib.jumpy_moi_kind(handle) == KINDS[name], f"Wrong kind for {name}")
    require(lib.jumpy_moi_bound(nodes["set"]) == 4.0, "Wrong LessThan bound")
    require(lib.jumpy_moi_arity(nodes["expression"]) == 2, "Wrong function arity")
    lib.jumpy_moi_collect()
    require(lib.jumpy_moi_bound(nodes["set"]) == 4.0, "Set did not survive GC")
    return nodes


def release_nodes(lib, nodes):
    # The expression itself must keep its children alive after their handles
    # are released from the registry.
    for name in ("variable", "constant"):
        require(lib.jumpy_moi_release(nodes[name]) == 0, f"Could not release {name}")
    lib.jumpy_moi_collect()
    require(lib.jumpy_moi_arity(nodes["expression"]) == 2, "Function lost its args")
    for name in ("expression", "set"):
        require(lib.jumpy_moi_release(nodes[name]) == 0, f"Could not release {name}")
    require(lib.jumpy_moi_release(nodes["set"]) == -1, "Released handle accepted")
    require(lib.jumpy_moi_release(0) == -1, "Invalid handle accepted")
    lib.jumpy_moi_collect()


def juliacall_runtime():
    import juliacall

    jl = juliacall.Main
    jl.seval("import MathOptInterface as MOI")
    return jl, juliacall.CONFIG["lib"]


def module_registered(jl):
    return bool(jl.seval("isdefined(Main, :JuMPyMOIProbe)"))


def julia_type_tokens(jl):
    return tuple(
        int(value)
        for value in jl.seval(
            "map(T -> UInt(pointer_from_objref(T)), "
            "(MOI.LessThan{Float64}, MOI.VariableIndex, "
            "Float64, MOI.ScalarNonlinearFunction))"
        )
    )


def library_type_tokens(lib):
    tokens = tuple(lib.jumpy_moi_type_token(kind) for kind in range(1, 5))
    require(all(tokens), "Library returned a null type token")
    return tokens


def standalone(path):
    lib = load_library(path)
    initialize(lib)
    bind_constructors(lib)
    nodes = make_nodes(lib)
    release_nodes(lib, nodes)
    report("standalone", status="passed", constructors=4, gc_and_release="passed")


def shared_sysimage(path):
    jl, runtime = juliacall_runtime()
    require(module_registered(jl), "JuliaCall did not load the constructor image")
    lib = load_library(path)
    require(
        address(lib.jl_is_initialized) == address(runtime.jl_is_initialized),
        "Constructor library and JuliaCall resolved different runtimes",
    )
    bind_constructors(lib)
    require(
        library_type_tokens(lib) == julia_type_tokens(jl),
        "MOI type identities differ despite using the constructor system image",
    )
    nodes = make_nodes(lib)
    solve = jl.seval(
        """
        begin
            import HiGHS
            function probe_constructor_solve(sh, vh, ch, fh)
                probe = Main.JuMPyMOIProbe
                s, x, c, f = map(probe.get_object, UInt64.((sh, vh, ch, fh)))
                @assert typeof(s) === MOI.LessThan{Float64}
                @assert typeof(x) === MOI.VariableIndex
                @assert typeof(c) === Float64
                @assert typeof(f) === MOI.ScalarNonlinearFunction
                @assert f.head === :+
                @assert f.args[1] === x && f.args[2] === c
                model = HiGHS.Optimizer()
                MOI.set(model, MOI.Silent(), true)
                @assert MOI.add_variable(model) == x
                affine = MOI.Nonlinear.SymbolicAD.simplify(f)
                MOI.Utilities.normalize_and_add_constraint(model, affine, s)
                objective = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
                MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
                MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
                MOI.optimize!(model)
                @assert MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
                value = MOI.get(model, MOI.VariablePrimal(), x)
                @assert isapprox(value, 3.0)
                return value
            end
        end
        """
    )
    value = float(solve(*(nodes[name] for name in KINDS)))
    borrow = jl.seval("h -> Main.JuMPyMOIProbe.get_object(UInt64(h))")
    retained_set = borrow(nodes["set"])
    release_nodes(lib, nodes)
    jl.GC.gc()
    require(retained_set.upper == 4.0, "JuliaCall did not root the borrowed MOI object")
    report(
        "shared-sysimage",
        status="passed",
        same_moi_types=True,
        native_objects_used_by_highs=True,
        optimum=value,
        images=1,
    )


def second_image(path):
    jl, runtime = juliacall_runtime()
    require(not module_registered(jl), "Default runtime already contains the probe")
    lib = load_library(path)
    bind_runtime(lib)
    require(
        address(lib.jl_is_initialized) == address(runtime.jl_is_initialized),
        "Expected one non-private runtime; refusing the second-image probe",
    )
    require(lib.jl_is_initialized() == 1, "Expected an already-initialized runtime")
    lib.jl_init_with_image_handle(lib._handle)
    registered = module_registered(jl)
    # Never invoke second-image Julia code just because dlopen succeeded.
    require(
        not registered,
        "Second image unexpectedly registered: investigate before invoking its exports",
    )
    report(
        "second-image",
        status="expected limitation confirmed",
        module_registered=registered,
        constructor_exports_called=False,
        live_objects_transferred=False,
    )


def second_compiled_image(path, backend_path):
    backend = load_library(backend_path)
    initialize(backend)
    backend.jl_eval_string.argtypes = [ctypes.c_char_p]
    backend.jl_eval_string.restype = ctypes.c_void_p
    backend.jl_exception_occurred.argtypes = []
    backend.jl_exception_occurred.restype = ctypes.c_void_p
    backend.jl_unbox_bool.argtypes = [ctypes.c_void_p]
    backend.jl_unbox_bool.restype = ctypes.c_uint8

    def registered(module):
        # Evaluate and unbox in the owning backend runtime only. In particular,
        # no object produced by the second image is ever dereferenced here.
        result = backend.jl_eval_string(f"isdefined(Main, :{module})".encode())
        require(
            not backend.jl_exception_occurred(),
            f"Julia exception checking whether {module} is registered",
        )
        require(result is not None, "Julia module check returned a null result")
        return bool(backend.jl_unbox_bool(result))

    require(registered("JuMPyHiGHS"), "Backend image did not register JuMPyHiGHS")
    require(not registered("JuMPyMOIProbe"), "Backend already contains the probe")
    constructors = load_library(path)
    bind_runtime(constructors)
    require(
        address(constructors.jl_is_initialized) == address(backend.jl_is_initialized),
        "Expected one non-private runtime; refusing the two-image probe",
    )
    constructors.jl_init_with_image_handle(constructors._handle)
    require(
        not registered("JuMPyMOIProbe"),
        "Second image unexpectedly registered: investigate before invoking its exports",
    )
    report(
        "second-compiled-image",
        status="expected limitation confirmed",
        backend_image_registered=True,
        constructor_image_registered=False,
        constructor_exports_called=False,
        live_objects_transferred=False,
    )


def private(path):
    jl, runtime = juliacall_runtime()
    require(not module_registered(jl), "Default runtime already contains the probe")
    public_tokens = julia_type_tokens(jl)
    lib = load_library(path)
    bind_runtime(lib)
    require(
        address(lib.jl_is_initialized) != address(runtime.jl_is_initialized),
        "Library is not isolated: refusing to call its constructor exports",
    )
    require(lib.jl_is_initialized() == 0, "Private runtime was already initialized")
    initialize(lib)
    bind_constructors(lib)
    private_tokens = library_type_tokens(lib)
    require(
        all(a != b for a, b in zip(public_tokens, private_tokens, strict=True)),
        "Expected distinct type identities in the isolated runtime",
    )
    nodes = make_nodes(lib)
    release_nodes(lib, nodes)
    require(not module_registered(jl), "Private module leaked into the public runtime")
    report(
        "private",
        status="passed",
        runtimes=2,
        same_moi_types=False,
        primitive_accessors="passed",
        foreign_addresses_dereferenced=False,
        live_objects_transferred=False,
    )


def run_child(mode, library, timeout, backend_library=None):
    env = os.environ.copy()
    env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
    env["PYTHON_JULIACALL_THREADS"] = "1"
    env["PYTHON_JULIACALL_INIT"] = "yes"
    if mode == "shared-sysimage":
        env["PYTHON_JULIACALL_SYSIMAGE"] = str(library)
    else:
        env.pop("PYTHON_JULIACALL_SYSIMAGE", None)
    command = [sys.executable, str(Path(__file__).resolve()), mode, str(library), "--worker"]
    if backend_library is not None:
        command.extend(("--backend-library", str(backend_library)))
    child = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
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
    parser.add_argument(
        "--backend-library",
        type=Path,
        help="Untrimmed libjumpy_highs; adds second-compiled-image to the all matrix",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    backend_library = (
        None if args.backend_library is None else args.backend_library.resolve(strict=True)
    )
    if args.mode == "second-compiled-image" and backend_library is None:
        parser.error("second-compiled-image requires --backend-library")
    if args.worker:
        require(args.mode != "all", "Workers run exactly one mode")
        if args.mode == "second-compiled-image":
            second_compiled_image(library, backend_library)
        else:
            globals()[args.mode.replace("-", "_")](library)
        return 0
    if args.mode == "all":
        modes = MODES[:3] if backend_library is None else MODES[:4]
    else:
        modes = (args.mode,)
    results = [run_child(mode, library, args.timeout, backend_library) for mode in modes]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
