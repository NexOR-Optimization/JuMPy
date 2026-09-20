"""Pure-Python safety checks for shared images and native object shutdown."""

import builtins
import ctypes
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from jumpy import backend
from jumpy.moi import MOIConstructors


class FakeJulia:
    def __init__(self, has_constructor_abi=True):
        self.has_constructor_abi = has_constructor_abi
        self.evaluations = []

    def seval(self, expression):
        self.evaluations.append(expression)
        return self.has_constructor_abi


def _fake_library(path, *, trimmed=False):
    return SimpleNamespace(
        _name=str(path),
        _jumpy_trimmed=trimmed,
        jumpy_runtime_image_path=lambda: os.fsencode(path),
        # Actual C-callable addresses let the production identity check run
        # without mocking ctypes.cast or initializing any Julia runtime.
        jl_is_initialized=ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 1),
    )


def _mock_juliacall(monkeypatch, runtime):
    juliacall = ModuleType("juliacall")
    juliacall.CONFIG = {"lib": runtime}
    monkeypatch.setitem(sys.modules, "juliacall", juliacall)


def test_trimmed_image_is_rejected_before_importing_juliacall(tmp_path, monkeypatch):
    lib = _fake_library(tmp_path / "libjumpy.so", trimmed=True)
    jl = FakeJulia()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "juliacall" or name.startswith("juliacall."):
            pytest.fail("A trimmed image must be rejected before importing JuliaCall")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="trimmed"):
        backend._require_shared_image(lib)
    with pytest.raises(RuntimeError, match="trimmed"):
        backend._check_shared_runtime(lib, jl)
    assert jl.evaluations == []


def test_untrimmed_image_can_be_shared(tmp_path):
    backend._require_shared_image(_fake_library(tmp_path / "libjumpy.so"))


def test_active_image_matches_canonical_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lib = _fake_library(tmp_path / "libjumpy.so")
    lib.jumpy_runtime_image_path = lambda: b"./libjumpy.so"
    backend._check_active_image(lib)


@pytest.mark.parametrize("image_is_symlink", [False, True])
def test_active_image_matches_symlink(tmp_path, image_is_symlink):
    target = tmp_path / "libjumpy.so"
    target.touch()
    symlink = tmp_path / "libjumpy-link.so"
    symlink.symlink_to(target)
    lib = _fake_library(target if image_is_symlink else symlink)
    active = symlink if image_is_symlink else target
    lib.jumpy_runtime_image_path = lambda: os.fsencode(active)
    backend._check_active_image(lib)


def test_different_active_image_is_rejected(tmp_path):
    lib = _fake_library(tmp_path / "libjumpy.so")
    lib.jumpy_runtime_image_path = lambda: os.fsencode(tmp_path / "another-image.so")
    with pytest.raises(RuntimeError, match="different system image"):
        backend._check_active_image(lib)


def test_missing_active_image_is_rejected(tmp_path):
    lib = _fake_library(tmp_path / "libjumpy.so")
    lib.jumpy_runtime_image_path = lambda: None
    with pytest.raises(RuntimeError, match="different system image"):
        backend._check_active_image(lib)


def test_matching_runtime_and_image_are_accepted(tmp_path, monkeypatch):
    lib = _fake_library(tmp_path / "libjumpy.so")
    _mock_juliacall(monkeypatch, lib)
    jl = FakeJulia()
    backend._check_shared_runtime(lib, jl)
    assert len(jl.evaluations) == 1
    assert "JuMPyMOIABI" in jl.evaluations[0]


def test_runtime_mismatch_is_rejected_before_image_or_julia_checks(tmp_path, monkeypatch):
    lib = _fake_library(tmp_path / "libjumpy.so")
    other_runtime = _fake_library(tmp_path / "libjumpy.so")
    _mock_juliacall(monkeypatch, other_runtime)
    jl = FakeJulia()

    def unexpected_image_lookup():
        pytest.fail("Image inspection must follow the runtime identity check")

    lib.jumpy_runtime_image_path = unexpected_image_lookup
    with pytest.raises(RuntimeError, match="different Julia runtimes"):
        backend._check_shared_runtime(lib, jl)
    assert jl.evaluations == []


def test_image_mismatch_is_rejected_before_julia_evaluation(tmp_path, monkeypatch):
    lib = _fake_library(tmp_path / "libjumpy.so")
    _mock_juliacall(monkeypatch, lib)
    lib.jumpy_runtime_image_path = lambda: os.fsencode(tmp_path / "another-image.so")
    jl = FakeJulia()
    with pytest.raises(RuntimeError, match="different system image"):
        backend._check_shared_runtime(lib, jl)
    assert jl.evaluations == []


def test_image_without_constructor_module_is_rejected(tmp_path, monkeypatch):
    lib = _fake_library(tmp_path / "libjumpy.so")
    _mock_juliacall(monkeypatch, lib)
    jl = FakeJulia(has_constructor_abi=False)
    with pytest.raises(RuntimeError, match="constructor ABI"):
        backend._check_shared_runtime(lib, jl)
    assert len(jl.evaluations) == 1


@pytest.mark.parametrize("shutdown", [False, True])
def test_native_objects_and_models_are_safe_to_close_at_shutdown(monkeypatch, shutdown):
    monkeypatch.setattr(backend, "_SHUTTING_DOWN", False)
    calls = []

    def release(handle):
        calls.append(("object", handle))
        return 0

    def free_model(handle):
        calls.append(("model", handle))
        return 0

    lib = SimpleNamespace(
        jumpy_moi_constant=lambda value: 42,
        jumpy_moi_release=release,
        jumpy_new_model=lambda: 99,
        jumpy_free_model=free_model,
    )
    node = MOIConstructors(lib).constant(1.0)
    ops = backend.JuliacOps(lib)
    if shutdown:
        backend._shutdown()
        assert backend._SHUTTING_DOWN
    node.close()
    ops.free()
    # Repeated cleanup neither re-enters Julia nor revives a released handle.
    node.close()
    ops.free()
    assert calls == ([] if shutdown else [("object", 42), ("model", 99)])
    assert ops._m is None
    with pytest.raises(RuntimeError, match="closed"):
        _ = node.handle


def test_runtime_cannot_be_initialized_during_shutdown(monkeypatch):
    monkeypatch.setattr(backend, "_SHUTTING_DOWN", True)

    def unexpected_open(path):
        pytest.fail("Shutdown must be checked before opening a library")

    monkeypatch.setattr(backend, "_open_lib", unexpected_open)
    with pytest.raises(RuntimeError, match="shutting down"):
        backend._init_lib("unused.so")


def test_cached_library_cannot_be_used_during_shutdown(monkeypatch):
    monkeypatch.setattr(backend, "_LIB", object())
    monkeypatch.setattr(backend, "_SHUTTING_DOWN", True)
    with pytest.raises(RuntimeError, match="shutting down"):
        backend._load_lib()


def test_factories_and_handles_cannot_enter_julia_after_shutdown(monkeypatch):
    monkeypatch.setattr(backend, "_SHUTTING_DOWN", False)
    lib = SimpleNamespace(jumpy_moi_constant=lambda value: 42)
    factory = MOIConstructors(lib)
    node = factory.constant(1.0)
    backend._shutdown()
    for operation in (
        lambda: MOIConstructors(lib),
        lambda: factory.scalar_set("<=", 1.0),
        lambda: factory.vector_set("==", 1),
        lambda: factory.constant(1.0),
        lambda: factory.variable(1),
        lambda: factory.scalar_nonlinear("+", []),
        lambda: factory.from_julia(None),
        lambda: node.handle,
        node.to_julia,
    ):
        with pytest.raises(RuntimeError, match="shutting down"):
            operation()
    node.close()  # cleanup must still be safe


@pytest.mark.parametrize("project_is_file", [False, True])
@pytest.mark.parametrize("use_xoptions", [False, True])
def test_shared_configuration_respects_selected_project(
    tmp_path, monkeypatch, project_is_file, use_xoptions,
):
    from jumpy import bridge_juliacall as bridge

    # The configurator writes new environment keys as well as modifying old
    # ones. Isolate the entire mapping, including keys absent at test start.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    project_dir = tmp_path / "environment"
    project_dir.mkdir()
    project_file = project_dir / "Project.toml"
    project_file.touch()
    selected = project_file if project_is_file else project_dir
    exe = tmp_path / "bin" / "julia"
    runtime = tmp_path / "libjulia.so"
    lib = _fake_library(tmp_path / "libjumpy.so")
    lib.jl_ver_string = lambda: b"1.13.0"
    lib.jumpy_runtime_library_path = lambda: os.fsencode(runtime)
    evaluations = []

    def evaluate(code):
        evaluations.append(code.decode())
        return 1

    lib.jl_eval_string = evaluate
    monkeypatch.delitem(sys.modules, "juliacall", raising=False)
    for key in ("EXE", "PROJECT", "LIB", "SYSIMAGE"):
        monkeypatch.delenv("PYTHON_JULIACALL_" + key, raising=False)
        monkeypatch.delitem(sys._xoptions, "juliacall-" + key.lower(), raising=False)
    if use_xoptions:
        monkeypatch.setitem(sys._xoptions, "juliacall-exe", str(exe))
        monkeypatch.setitem(sys._xoptions, "juliacall-project", str(selected))
        # -X is authoritative over environment settings, as in JuliaCall.
        monkeypatch.setenv("PYTHON_JULIACALL_EXE", "/unused/julia")
        monkeypatch.setenv("PYTHON_JULIACALL_PROJECT", "/unused/environment")
    else:
        monkeypatch.setenv("PYTHON_JULIACALL_EXE", str(exe))
        monkeypatch.setenv("PYTHON_JULIACALL_PROJECT", str(selected))

    def installation(command, **kwargs):
        assert command[0] == str(exe)
        return SimpleNamespace(stdout="1.13.0\0/matching/bin\0/matching/stdlib")

    monkeypatch.setattr(bridge.subprocess, "run", installation)
    bridge._configure_shared(lib)
    assert os.environ["PYTHON_JULIACALL_PROJECT"] == str(selected)
    assert os.environ["PYTHON_JULIACALL_LIB"] == str(runtime)
    assert os.environ["PYTHON_JULIACALL_SYSIMAGE"] == lib._name
    assert len(evaluations) == 1
    assert str(project_file) in evaluations[0]
    assert "Project.toml/Project.toml" not in evaluations[0]
    assert 'Sys.STDLIB = "/matching/stdlib"' in evaluations[0]
