"""
Tests for compiled-library discovery in jumpy.backend.

Pure Python: only the error paths are exercised, so neither Julia nor the
compiled library is needed.
"""

import os
import sys
sys.path.insert(0, "src")

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


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {test.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
