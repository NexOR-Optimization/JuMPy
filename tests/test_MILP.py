"""
Test implementation of binary and integer variables by checking the mof output
"""

import json
import jumpy as jp

def test_binary_variables_stored_correctly():
    """VariableBlock records binary=True."""
    m = jp.Model()
    x = m.variables(3, binary=True, name="x")
    assert m._var_blocks[-1].binary is True
    assert m._var_blocks[-1].integer is False

def test_integer_variables_stored_correctly():
    m = jp.Model()
    x = m.variables(3, integer=True, name="x")
    assert m._var_blocks[-1].integer is True
    assert m._var_blocks[-1].binary is False

def test_single_binary_variable():
    m = jp.Model()
    x = m.variable(binary=True, name="x")
    assert m._var_blocks[-1].binary is True

def test_default_is_false():
    """Existing code not broken, continuous variables unchanged."""
    m = jp.Model()
    x = m.variables(3, lower=0, name="x")
    assert m._var_blocks[-1].binary is False
    assert m._var_blocks[-1].integer is False

if __name__ == "__main__":
    import tempfile, sys, pathlib, traceback

    passed = failed = 0
    tmp = pathlib.Path(tempfile.mkdtemp())
    tests = [v for k, v in globals().items() if k.startswith("test_")]

    for test in tests:
        try:
            if "tmp_path" in test.__code__.co_varnames:
                test(tmp)
            else:
                test()
            passed += 1
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)