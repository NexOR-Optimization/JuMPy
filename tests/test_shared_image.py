"""Run real image-sharing checks in disposable Python processes.

Build a library with julia/build.jl, or select one using JUMPY_LIB. Each worker
starts fresh; even reading the image profile happens outside the pytest process.
Compiled-only workers use ``python -S`` so JuliaCall is not available to them.
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


_RESULT_PREFIX = "JUMPY_SHARED_TEST="


def _collect_native(lib):
    # Julia's runtime GC API works without eval, including in trimmed images.
    collect = lib.jl_gc_collect
    collect.argtypes = [ctypes.c_int]
    collect.restype = None
    gc.collect()
    collect(1)  # JL_GC_FULL


def _solve_compiled(set_object, bound):
    import jumpy as jp

    model = jp.Model(backend="juliac")
    try:
        x, y = model.variables(2, lower=0)
        expression = x + y
        ops = model._ops
        index = ops._lib.jumpy_add_constraint_set(
            ops._m, expression.moi, set_object.handle,
        )
        assert index >= 0, "The compiled model rejected the native set"
        model.objective = jp.maximize(expression)
        model.optimize()
        optimum = model.value(x) + model.value(y)
        assert math.isclose(optimum, bound, abs_tol=1e-7), optimum
        return optimum
    finally:
        model.close()


def _solve_juliacall(jl, native_set, bound, model=None):
    import jumpy as jp

    if model is None:
        model = jp.Model(backend="juliacall")
    try:
        x, y = model.variables(2, lower=0)
        expression = x + y
        # Deliberately exercise the actual Julia object. The public two-argument
        # Model.constraint interface is a separate, later change.
        jl.MOI.Utilities.normalize_and_add_constraint(
            model._ops._optimizer,
            model._ops._simplify(expression.moi),
            native_set,
        )
        model.objective = jp.maximize(expression)
        model.optimize()
        optimum = model.value(x) + model.value(y)
        assert math.isclose(optimum, bound, abs_tol=1e-7), optimum
        return optimum
    finally:
        model.close()


def _compiled_worker():
    from jumpy.moi import MOIConstructors

    assert "juliacall" not in sys.modules
    constructors = MOIConstructors()
    lib = constructors._lib
    with constructors.scalar_set("<=", 1.0) as set_object:
        _collect_native(lib)
        assert lib.jumpy_moi_kind(set_object.handle) == 1
        optimum = _solve_compiled(set_object, 1.0)
        released_handle = set_object.handle
    assert lib.jumpy_moi_kind(released_handle) == -1

    x = constructors.variable(1)
    constant = constructors.constant(2.0)
    function = constructors.scalar_nonlinear("+", [x, constant])
    x.close()
    constant.close()
    try:
        _collect_native(lib)
        assert lib.jumpy_moi_kind(function.handle) == 11
    finally:
        function.close()
    assert "juliacall" not in sys.modules
    return {"optimum": optimum, "juliacall_imported": False}


def _shared_worker(startup):
    import jumpy as jp
    from jumpy.bridge_juliacall import _julia
    from jumpy.moi import MOIConstructors

    assert "juliacall" not in sys.modules
    if startup == "compiled-first-project-file":
        executable = os.environ.get("PYTHON_JULIACALL_EXE")
        project = os.environ.get("PYTHON_JULIACALL_PROJECT")
        if executable is None or project is None:
            import juliapkg

            executable = executable or juliapkg.executable()
            project = project or juliapkg.project()
        project = Path(project)
        if project.is_dir():
            project = project / "Project.toml"
        os.environ["PYTHON_JULIACALL_EXE"] = str(executable)
        os.environ["PYTHON_JULIACALL_PROJECT"] = str(project)
    initial_model = None
    if startup == "juliacall-first":
        initial_model = jp.Model(backend="juliacall")
        assert "juliacall" in sys.modules
    constructors = MOIConstructors()
    set_object = constructors.scalar_set("<=", 1.0)
    if startup.startswith("compiled-first"):
        assert "juliacall" not in sys.modules
    native_set = set_object.to_julia()
    jl = _julia()
    try:
        assert bool(jl.seval(
            "Main.JuMPyMOI === Main.JuMPyHiGHS.JuMPyMOI === "
            "Main.JuMPyHiGHS.JuMPyMOIABI.JuMPyMOI"
        )), "The two interfaces must reuse the compiled constructor module"
        is_same = jl.seval(
            "(value, handle) -> value === "
            "JuMPyHiGHS.JuMPyMOIABI.get_object(UInt64(handle))"
        )
        assert bool(is_same(native_set, set_object.handle))
        assert float(native_set.upper) == 1.0
        compiled_optimum = _solve_compiled(set_object, 1.0)
        juliacall_optimum = _solve_juliacall(jl, native_set, 1.0, initial_model)
        initial_model = None
    finally:
        set_object.close()
        if initial_model is not None:
            initial_model.close()

    # JuliaCall roots the borrowed object after the C registry releases it.
    _collect_native(constructors._lib)
    assert float(native_set.upper) == 1.0
    _solve_juliacall(jl, native_set, 1.0)

    # The reverse direction retains the native Julia object, not a serialized
    # copy or a Python description of the set.
    native_reverse = jl.MOI.LessThan(2.0)
    with constructors.from_julia(native_reverse) as reverse:
        assert bool(is_same(native_reverse, reverse.handle))
        reverse_optimum = _solve_compiled(reverse, 2.0)

    # Native nonlinear functions retain their children even after the child
    # handles are explicitly released, and can also be borrowed by JuliaCall.
    with constructors.variable(1) as x, constructors.constant(2.0) as constant:
        function = constructors.scalar_nonlinear("+", [x, constant])
    try:
        _collect_native(constructors._lib)
        native_function = function.to_julia()
        assert bool(jl.seval(
            "f -> f isa MOI.ScalarNonlinearFunction && "
            "f.head === :+ && f.args == Any[MOI.VariableIndex(1), 2.0]"
        )(native_function))
    finally:
        function.close()
    _collect_native(constructors._lib)
    assert bool(jl.seval("f -> length(f.args) == 2")(native_function))
    return {
        "startup": startup,
        "compiled_optimum": compiled_optimum,
        "juliacall_optimum": juliacall_optimum,
        "reverse_optimum": reverse_optimum,
        "borrowed_objects_survive_release": True,
    }


def _trimmed_worker():
    from jumpy.moi import MOIConstructors

    assert "juliacall" not in sys.modules
    constructors = MOIConstructors()
    assert constructors._lib._jumpy_trimmed
    with constructors.scalar_set("<=", 1.0) as set_object:
        try:
            set_object.to_julia()
        except RuntimeError as error:
            assert "trimmed" in str(error).lower(), str(error)
        else:
            raise AssertionError("A trimmed image unexpectedly initialized JuliaCall")
    assert "juliacall" not in sys.modules
    return {"rejected": True, "juliacall_imported": False}


def _wrong_image_worker():
    # Import JuliaCall itself, not the JuMPy loader: it must start with the
    # ordinary Julia image and cannot later attach an independent compiled one.
    from juliacall import Main as jl
    from jumpy import backend
    from jumpy.moi import MOIConstructors

    assert not bool(jl.seval("isdefined(Main, :JuMPyHiGHS)"))
    original_open = backend._open_lib
    invoked_exports = []

    def forbidden_export(*args):
        invoked_exports.append(True)
        raise AssertionError("Called a Julia export from an inactive image")

    def guarded_open(path):
        lib = original_open(path)
        # If the image guard regresses, fail without executing foreign-image
        # Julia code. Image metadata and identity functions remain callable.
        lib.jl_init_with_image_handle = forbidden_export
        lib.jumpy_moi_scalar_set = forbidden_export
        lib.jumpy_new_model = forbidden_export
        return lib

    backend._open_lib = guarded_open
    try:
        MOIConstructors()
    except RuntimeError as error:
        assert "different system image" in str(error), str(error)
    else:
        raise AssertionError("An incompatible active image was accepted")
    assert not invoked_exports
    return {"rejected": True, "native_exports_called": False}


def _worker(mode):
    if mode == "profile":
        from jumpy import backend

        lib = backend._open_lib(os.environ["JUMPY_LIB"])
        assert not lib.jl_is_initialized()
        return {"profile": "trimmed" if lib._jumpy_trimmed else "shared"}
    if mode == "compiled-only":
        return _compiled_worker()
    if mode in ("compiled-first", "compiled-first-project-file", "juliacall-first"):
        return _shared_worker(mode)
    if mode == "trimmed-rejection":
        return _trimmed_worker()
    if mode == "wrong-image":
        return _wrong_image_worker()
    raise ValueError(f"Unknown worker mode: {mode}")


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker"
    result = _worker(sys.argv[2])
    print(_RESULT_PREFIX + json.dumps(result), flush=True)
    raise SystemExit(0)


# Kept below the worker entry point: python -S intentionally cannot import
# pytest or JuliaCall, but it can import JuMPy's dependency-free Python package.
import pytest
import jumpy


def _run_worker(mode, library, directory, *, no_site=False):
    env = os.environ.copy()
    env["JUMPY_LIB"] = str(library)
    # Test automatic image selection, even when the parent test process has
    # already initialized JuliaCall and populated these environment variables.
    env.pop("PYTHON_JULIACALL_SYSIMAGE", None)
    env.pop("PYTHON_JULIACALL_LIB", None)
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")
    package_root = str(Path(jumpy.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend([str(Path(__file__).resolve()), "--worker", mode])
    completed = subprocess.run(
        command, cwd=directory, env=env, capture_output=True, text=True, timeout=120,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert completed.returncode == 0, (
        f"{mode} worker exited with {completed.returncode}:\n{output}"
    )
    records = [
        line[len(_RESULT_PREFIX):]
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(records) == 1, f"Missing worker result:\n{output}"
    return json.loads(records[0])


@pytest.fixture(scope="module")
def compiled_image(tmp_path_factory):
    configured = jumpy.backend.find_lib()
    if configured is None:
        pytest.skip("Build a compiled library or select one with JUMPY_LIB")
    library = Path(configured).resolve()
    assert library.is_file(), f"JUMPY_LIB does not exist: {library}"
    directory = tmp_path_factory.mktemp("shared-image")
    profile = _run_worker("profile", library, directory, no_site=True)["profile"]
    return library, directory, profile


def test_compiled_constructors_work_without_site_packages(compiled_image):
    library, directory, _ = compiled_image
    result = _run_worker("compiled-only", library, directory, no_site=True)
    assert result["optimum"] == pytest.approx(1.0)
    assert result["juliacall_imported"] is False


@pytest.mark.parametrize(
    "startup", ["compiled-first", "compiled-first-project-file", "juliacall-first"],
)
def test_native_objects_work_in_both_backends(compiled_image, startup):
    library, directory, profile = compiled_image
    if profile == "trimmed":
        pytest.skip("Shared-runtime execution requires an untrimmed image")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("JuliaCall is not installed")
    result = _run_worker(startup, library, directory)
    assert result["startup"] == startup
    assert result["compiled_optimum"] == pytest.approx(1.0)
    assert result["juliacall_optimum"] == pytest.approx(1.0)
    assert result["reverse_optimum"] == pytest.approx(2.0)
    assert result["borrowed_objects_survive_release"] is True


def test_trimmed_image_rejects_juliacall_before_import(compiled_image):
    library, directory, profile = compiled_image
    if profile != "trimmed":
        pytest.skip("This check requires a trimmed image")
    result = _run_worker("trimmed-rejection", library, directory, no_site=True)
    assert result["rejected"] is True
    assert result["juliacall_imported"] is False


def test_incompatible_active_image_is_rejected_before_native_calls(compiled_image):
    library, directory, profile = compiled_image
    if profile == "trimmed":
        pytest.skip("JuliaCall startup is tested with the shared build profile")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("JuliaCall is not installed")
    result = _run_worker("wrong-image", library, directory)
    assert result["rejected"] is True
    assert result["native_exports_called"] is False
