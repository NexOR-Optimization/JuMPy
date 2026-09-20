"""Backend selection by imports, without starting a Julia runtime."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import jumpy
import jumpy.highs as highs
import jumpy.juliacall as juliacall
from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


def test_importing_either_facade_does_not_load_julia(tmp_path):
    # Start fresh so prior tests cannot hide an eager import. -S also proves
    # both modules can be imported without JuliaCall or other site packages.
    script = """
import builtins
import ctypes
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    assert name != 'juliacall' and not name.startswith('juliacall.'), name
    return original_import(name, *args, **kwargs)
def unexpected_load(*args, **kwargs):
    raise AssertionError('Importing a facade must not load a native library')
builtins.__import__ = checked_import
ctypes.CDLL = unexpected_load
import jumpy.highs as highs
import jumpy.juliacall as juliacall
assert 'juliacall' not in sys.modules
assert highs.Model is not juliacall.Model
assert 'MOI' in highs.__all__ and 'MOI' in juliacall.__all__
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(jumpy.__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], env=env, cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_module_has_no_implicit_model_backend():
    assert "Model" not in jumpy.__all__
    assert not hasattr(jumpy, "Model")


@pytest.mark.parametrize("facade", [highs, juliacall])
def test_facades_reexport_common_modeling_helpers(facade):
    for name in jumpy.__all__:
        assert getattr(facade, name) is getattr(jumpy, name)


def test_highs_model_selects_compiled_ops(monkeypatch):
    lib = object()
    ops = MockOps()
    calls = []
    monkeypatch.setattr(backend, "_load_lib", lambda: lib)

    def create_ops(selected_lib):
        calls.append(selected_lib)
        return ops

    monkeypatch.setattr(backend, "JuliacOps", create_ops)
    model = highs.Model()
    try:
        assert isinstance(model, Model)
        assert model._ops is ops
        assert calls == [lib]
    finally:
        model.close()


def test_juliacall_model_selects_juliacall_ops(monkeypatch):
    ops = MockOps()
    monkeypatch.setattr(bridge_juliacall, "JuliaCallOps", lambda: ops)
    model = juliacall.Model()
    try:
        assert isinstance(model, Model)
        assert model._ops is ops
    finally:
        model.close()


@pytest.mark.parametrize("facade", [highs, juliacall])
def test_facade_model_does_not_accept_backend_argument(facade):
    with pytest.raises(TypeError):
        facade.Model(backend="juliac")


def test_juliacall_moi_is_the_actual_lazy_julia_module(monkeypatch):
    native_moi = object()
    calls = []

    def initialize():
        calls.append(True)
        return SimpleNamespace(MOI=native_moi)

    monkeypatch.setattr(bridge_juliacall, "_julia", initialize)
    assert not calls
    assert juliacall.MOI is native_moi
    assert calls == [True]


def test_juliacall_unknown_attribute_does_not_initialize_julia(monkeypatch):
    def unexpected_init():
        pytest.fail("An unknown Python attribute must not initialize Julia")

    monkeypatch.setattr(bridge_juliacall, "_julia", unexpected_init)
    with pytest.raises(AttributeError):
        _ = juliacall.nonexistent


def test_compiled_backend_rejects_active_juliacall_before_loading_library(monkeypatch):
    fake = ModuleType("juliacall")
    fake.CONFIG = {"inited": True}
    monkeypatch.setitem(sys.modules, "juliacall", fake)

    def unexpected_load(*args, **kwargs):
        pytest.fail("An initialized JuliaCall runtime must be rejected before CDLL")

    monkeypatch.setattr(backend.ctypes, "CDLL", unexpected_load)
    with pytest.raises(RuntimeError, match="one JuMPy backend per process"):
        backend._init_lib("unused.so")


def test_juliacall_rejects_active_compiled_backend_before_import(monkeypatch):
    import builtins

    monkeypatch.setattr(backend, "_LIB", object())
    monkeypatch.setattr(bridge_juliacall, "_JL", None)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "juliacall":
            pytest.fail("An active compiled runtime must be rejected before JuliaCall import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    with pytest.raises(RuntimeError, match="one JuMPy backend per process"):
        bridge_juliacall._julia()
