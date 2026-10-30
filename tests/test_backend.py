"""
Tests for compiled-library discovery in jumpy.backend.

Pure Python: only the error paths are exercised, so neither Julia nor the
compiled library is needed.
"""

import os
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall


def _fresh_load(jumpy_lib):
    """Run _load_lib with a clean cache and JUMPY_LIB set to `jumpy_lib`."""
    saved_lib, saved_env = backend._LIB, os.environ.get("JUMPY_LIB")
    backend._LIB = None
    os.environ["JUMPY_LIB"] = jumpy_lib
    try:
        return backend._load_lib("highs")
    finally:
        backend._LIB = saved_lib
        if saved_env is None:
            del os.environ["JUMPY_LIB"]
        else:
            os.environ["JUMPY_LIB"] = saved_env


def test_jumpy_lib_missing_file_errors():
    """JUMPY_LIB pointing to a missing file must error, not silently fall
    back to another library (regression test for d1f1f4a)."""
    path = "/nonexistent/libjumpy_highs.so"
    try:
        _fresh_load(path)
    except FileNotFoundError as e:
        assert "JUMPY_LIB" in str(e)
        assert path in str(e)
    else:
        raise AssertionError("expected FileNotFoundError for missing JUMPY_LIB")


def test_jumpy_lib_directory_errors(tmp_path):
    """A directory is not a loadable library either; the existence check
    passes but CDLL must fail — never a silent fallback."""
    directory = tmp_path / "libjumpy_highs.so"
    directory.mkdir()
    try:
        _fresh_load(str(directory))
    except OSError:
        pass
    else:
        raise AssertionError("expected an error for JUMPY_LIB=/tmp")


@pytest.mark.parametrize(
    "missing", ["jumpy_apply", "jumpy_integer_constant", "jumpy_integer_iterator",
                "jumpy_variables", "jumpy_value", "jumpy_add_constraint", "jumpy_dual",
                "jumpy_objective_value",
                "jumpy_has_values", "jumpy_has_duals", "jumpy_termination_status",
                "jumpy_set_time_limit_sec", "jumpy_set_string_attribute",
                "jumpy_set_bool_attribute", "jumpy_set_int_attribute",
                "jumpy_set_float_attribute"],
)
def test_old_abi_is_rejected_before_julia_initialization(monkeypatch, missing):
    def unexpected_init(*args):
        pytest.fail("An incompatible ABI must be rejected before starting Julia")

    symbols = {
        name: lambda *args: None
        for name in (
            "jumpy_apply", "jumpy_integer_constant", "jumpy_integer_iterator",
            "jumpy_variables", "jumpy_value", "jumpy_add_constraint", "jumpy_dual",
            "jumpy_objective_value",
            "jumpy_has_values", "jumpy_has_duals", "jumpy_termination_status",
            "jumpy_less_than", "jumpy_greater_than", "jumpy_equal_to",
            "jumpy_zero_one", "jumpy_integer", "jumpy_free_set",
            "jumpy_set_time_limit_sec", "jumpy_set_string_attribute",
            "jumpy_set_bool_attribute", "jumpy_set_int_attribute",
            "jumpy_set_float_attribute",
        )
        if name != missing
    }
    lib = SimpleNamespace(**symbols, jl_init_with_image_handle=unexpected_init)
    monkeypatch.setattr(backend.ctypes, "CDLL", lambda *args, **kwargs: lib)
    with pytest.raises(RuntimeError, match="Rebuild.*native JuMP API"):
        backend._init_lib("old-library.so")


def test_juliacall_ops_wraps_an_existing_jump_model(monkeypatch):
    native_model = object()
    jl = SimpleNamespace(
        MOI=object(),
        JuMPyModel=object(),
        seval=lambda code: object(),
    )
    monkeypatch.setattr(bridge_juliacall, "_julia", lambda: jl)

    ops = bridge_juliacall.JuliaCallOps(native_model)
    assert ops._model is native_model
    assert ops._jl is jl
    assert ops._jump is jl.JuMPyModel


def test_nexor_library_paths():
    assert all("libjumpy_nexor" in path for path in backend._default_paths("nexor"))
    assert all("libjumpy_highs" in path for path in backend._default_paths("highs"))


def test_nexor_library_override(monkeypatch):
    monkeypatch.setenv("JUMPY_LIB", "/custom/libjumpy_nexor.so")
    assert backend.find_lib("nexor") == "/custom/libjumpy_nexor.so"


def test_jumpy_lib_only_overrides_its_own_backend(monkeypatch):
    monkeypatch.setenv("JUMPY_LIB", "/custom/libjumpy_highs.so")
    monkeypatch.setattr(backend, "_LIB", None)
    assert backend.find_lib("nexor") is None
    with pytest.raises(FileNotFoundError, match="jumpy.nexor needs libjumpy_nexor"):
        backend._load_lib("nexor")


@pytest.mark.parametrize(
    ("value", "setter", "sent"),
    [
        ("HiGHS", "jumpy_set_string_attribute", b"HiGHS"),
        (True, "jumpy_set_bool_attribute", True),
        (3, "jumpy_set_int_attribute", 3),
        (1e-6, "jumpy_set_float_attribute", 1e-6),
    ],
)
def test_compiled_set_attribute_selects_typed_setter(value, setter, sent):
    calls = []
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 1,
        jumpy_free_model=lambda m: 0,
        **{setter: lambda m, name, v: calls.append((m, name, v)) or 0},
    )
    ops = backend.JuliacOps(lib, None)
    ops.set_attribute("solver", value)
    assert calls == [(1, b"solver", sent)]
    assert type(calls[0][2]) is type(sent)
    ops.free()


def test_compiled_set_attribute_errors():
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 1,
        jumpy_free_model=lambda m: 0,
        jumpy_set_int_attribute=lambda m, name, v: -1,
    )
    ops = backend.JuliacOps(lib, None)
    with pytest.raises(RuntimeError, match="'threads'"):
        ops.set_attribute("threads", 4)
    ops.free()


def test_nexor_facade_uses_native_model_api(monkeypatch):
    import jumpy
    import jumpy.nexor as nexor
    from mock_ops import MockOps

    ops, lib, calls = MockOps(), object(), []
    monkeypatch.setattr(backend, "_load_lib", lambda name: calls.append(name) or lib)
    monkeypatch.setattr(
        backend, "JuliacOps", lambda selected, MOI: ops if selected is lib and MOI is nexor.MOI else None
    )
    m = nexor.Model()
    try:
        x = m.variables(2, lower=0)
        assert calls == ["nexor"]
        assert ops.constraints == [(("getindex", x.ref, k), ("GreaterThan", 0.0)) for k in (1, 2)]
        for name in jumpy.__all__:
            assert getattr(nexor, name) is getattr(jumpy, name)
    finally:
        m.close()


def test_nexor_sets_load_the_nexor_library(monkeypatch):
    import jumpy.nexor as nexor

    calls = []
    lib = SimpleNamespace(jumpy_less_than=lambda rhs: 42, jumpy_free_set=lambda handle: 0)
    monkeypatch.setattr(backend, "_load_lib", lambda name: calls.append(name) or lib)
    with nexor.MOI.LessThan(1.0) as set_:
        assert set_._lib is lib
        assert set_.handle == 42
    assert calls == ["nexor"]


def test_compiled_backends_cannot_mix_runtimes(monkeypatch):
    lib = object()
    monkeypatch.setattr(backend, "_LIB", lib)
    monkeypatch.setattr(backend, "_LIB_NAME", "nexor")
    assert backend._load_lib("nexor") is lib
    with pytest.raises(RuntimeError, match="jumpy.nexor is already loaded"):
        backend._load_lib("highs")
