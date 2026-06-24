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

def test_binary_exports_zeroone_to_mof(tmp_path):
    """MOF export emits ZeroOne set for binary variables."""
    m = jp.Model()
    x = m.variables(3, binary=True, name="x")
    m.objective = jp.minimize(x[0] + x[1] + x[2])
    path = str(tmp_path / "bin.mof.json")
    m.write_mof(path)
    doc = json.load(open(path))
    zeroone = [c for c in doc["constraints"]
               if c["set"]["type"] == "ZeroOne"]
    assert len(zeroone) == 3
    # remove the created file

def test_integer_exports_integer_to_mof(tmp_path):
    m = jp.Model()
    x = m.variables(2, integer=True, name="x")
    m.objective = jp.minimize(x[0] + x[1])
    path = str(tmp_path / "int.mof.json")
    m.write_mof(path)
    doc = json.load(open(path))
    integer_cons = [c for c in doc["constraints"]
                    if c["set"]["type"] == "Integer"]
    assert len(integer_cons) == 2

def test_continuous_variables_unaffected_by_binary_flag(tmp_path):
    """Continuous vars next to binary vars still export without ZeroOne."""
    m = jp.Model()
    x = m.variables(2, binary=True, name="x")
    y = m.variables(2, lower=0, name="y")      # continuous
    m.objective = jp.minimize(x[0] + y[0])
    path = str(tmp_path / "mixed.mof.json")
    m.write_mof(path)
    doc = json.load(open(path))
    zeroone = [c for c in doc["constraints"]
               if c["set"]["type"] == "ZeroOne"]
    assert len(zeroone) == 2    # only the x variables

def test_zeroone_variable_name_is_correct(tmp_path):
    """Each ZeroOne constraint references the correct variable."""
    m = jp.Model()
    x = m.variables(2, binary=True, name="x")
    m.objective = jp.minimize(x[0] + x[1])
    path = str(tmp_path / "names.mof.json")
    m.write_mof(path)
    doc = json.load(open(path))
    names = {c["function"]["name"]
             for c in doc["constraints"]
             if c["set"]["type"] == "ZeroOne"}
    assert names == {"x[0]", "x[1]"}

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