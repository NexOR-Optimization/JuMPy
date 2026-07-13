"""
Tests for eager expression building, using a mock ops object.

The mock records every MOI call as a tuple, so these tests check exactly
what a backend receives — pure Python, no Julia needed.
"""

import sys
sys.path.insert(0, "src")

from jumpy.expressions import Node, Parameter, Variable, VariableVector, sin


class MockOps:
    def __init__(self):
        self.constraints = []
        self.groups = []
        self.num_vars = 0

    def constant(self, v):
        return v

    def variable(self, index):
        return ("var", index)

    def scalar_nonlinear(self, head, args):
        return (head, *args)

    def iterator(self, values):
        return ("iterator", tuple(values))

    def contiguous_variables(self, start, count):
        return ("block", start, count)

    def float_array(self, values):
        return ("data", tuple(values))

    def add_variables(self, count):
        start = self.num_vars
        self.num_vars += count
        return start

    def add_constraint(self, func, sense, rhs):
        self.constraints.append((func, sense, rhs))

    def add_constraint_group(self, func, sense, linear):
        self.groups.append((func, sense, linear))


def test_arithmetic_builds_scalar_nonlinear():
    ops = MockOps()
    x = Variable(ops, 0)
    y = Variable(ops, 1)
    expr = x + 2 * y
    assert expr.moi == ("+", ("var", 0), ("*", 2.0, ("var", 1)))
    assert expr.linear


def test_reflected_and_unary_operators():
    ops = MockOps()
    x = Variable(ops, 0)
    assert (1 - x).moi == ("-", 1.0, ("var", 0))
    assert (-x).moi == ("-", ("var", 0))
    assert (2.0 / x).linear is False
    assert (x**2).linear is False


def test_nonlinear_functions():
    ops = MockOps()
    x = Variable(ops, 0)
    expr = sin(x) + 1
    assert expr.moi == ("+", ("sin", ("var", 0)), 1.0)
    assert not expr.linear


def test_comparison_normalizes():
    ops = MockOps()
    x = Variable(ops, 0)
    con = x + 1 <= 10
    assert con.sense == "<="
    assert con.func.moi == ("-", ("+", ("var", 0), 1.0), 10.0)


def test_variable_vector_concrete_indexing():
    ops = MockOps()
    start = ops.add_variables(3)
    x = VariableVector(ops, start, 3, "x")
    assert x[2].index == 2
    assert x[2].moi == ("var", 2)
    assert len(x) == 3
    assert [v.index for v in x] == [0, 1, 2]


def test_variable_vector_symbolic_indexing():
    ops = MockOps()
    x = VariableVector(ops, 0, 10, "x")
    i = Node(ops, ops.iterator([0.0, 1.0]))
    # 0-based Python index -> 1-based Julia index
    assert x[i].moi == (
        "getindex",
        ("block", 0, 10),
        ("+", ("iterator", (0.0, 1.0)), 1.0),
    )
    assert x[i].linear


def test_parameter_indexing():
    ops = MockOps()
    p = Parameter(ops, [1.0, 2.0, 3.0], "costs")
    assert p[1] == 2.0
    i = Node(ops, ops.iterator([0.0, 1.0]))
    assert p[i].moi == (
        "getindex",
        ("data", (1.0, 2.0, 3.0)),
        ("+", ("iterator", (0.0, 1.0)), 1.0),
    )


def test_template_linearity_flag():
    ops = MockOps()
    x = VariableVector(ops, 0, 10, "x")
    i = Node(ops, ops.iterator([0.0, 1.0]))
    assert (x[i] + x[i + 1] <= 10).func.linear
    assert not (sin(x[i]) <= 1).func.linear


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
