"""
Tests for compiled-library discovery in jumpy.backend.

Pure Python: only the error paths are exercised, so neither Julia nor the
compiled library is needed.
"""

import os
from types import SimpleNamespace

import pytest

from jumpy import backend


def _fresh_load(jumpy_lib):
    """Run _load_lib with a clean cache and JUMPY_LIB set to `jumpy_lib`."""
    saved_lib, saved_env = backend._LIB, os.environ.get("JUMPY_LIB")
    backend._LIB = None
    os.environ["JUMPY_LIB"] = jumpy_lib
    try:
        return backend._load_lib()
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


def test_jumpy_lib_directory_errors():
    """A directory is not a loadable library either; the existence check
    passes but CDLL must fail — never a silent fallback."""
    try:
        _fresh_load("/tmp")
    except OSError:
        pass
    else:
        raise AssertionError("expected an error for JUMPY_LIB=/tmp")


@pytest.mark.parametrize(
    "missing", ["jumpy_apply", "jumpy_integer_constant", "jumpy_integer_iterator",
                "jumpy_variables", "jumpy_value"],
)
def test_old_abi_is_rejected_before_julia_initialization(monkeypatch, missing):
    def unexpected_init(*args):
        pytest.fail("An incompatible ABI must be rejected before starting Julia")

    symbols = {
        name: lambda *args: None
        for name in (
            "jumpy_apply", "jumpy_integer_constant", "jumpy_integer_iterator",
            "jumpy_variables", "jumpy_value",
            "jumpy_less_than", "jumpy_greater_than", "jumpy_equal_to",
            "jumpy_zero_one", "jumpy_integer", "jumpy_free_set",
        )
        if name != missing
    }
    lib = SimpleNamespace(**symbols, jl_init_with_image_handle=unexpected_init)
    monkeypatch.setattr(backend.ctypes, "CDLL", lambda *args, **kwargs: lib)
    with pytest.raises(RuntimeError, match="Rebuild.*native JuMP API"):
        backend._init_lib("old-library.so")
