"""Tests for eager expression building, using the mock ops object."""

from jumpy.expressions import Node, Parameter, Variable, VariableVector, sin
from mock_ops import MockOps


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
