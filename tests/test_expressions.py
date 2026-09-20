"""Tests for eager expression building, using the mock ops object."""

import operator

import pytest

from jumpy.expressions import Node, Parameter, Variable, VariableVector, sin
from jumpy.model import Model
from mock_ops import MockOps


def test_arithmetic_forwards_native_operations():
    ops = MockOps()
    x = Variable(ops, 0)
    y = Variable(ops, 1)
    expr = x + 2 * y
    assert expr.ref == ("+", ("var", 0), ("*", 2, ("var", 1)))


def test_reflected_and_unary_operators():
    ops = MockOps()
    x = Variable(ops, 0)
    assert (1 - x).ref == ("-", 1, ("var", 0))
    assert (-x).ref == ("-", ("var", 0))
    assert (2.0 / x).ref == ("/", 2.0, ("var", 0))
    assert (x**2).ref == ("^", ("var", 0), 2)


@pytest.mark.parametrize("value", [2, 2.0])
@pytest.mark.parametrize("op", [operator.add, operator.mul, operator.pow])
def test_integer_and_float_literals_keep_their_types(value, op):
    x = Variable(MockOps(), 0)
    assert type(op(x, value).ref[-1]) is type(value)


@pytest.mark.parametrize("values", [range(3), [0.0, 0.5]])
def test_model_iterator_preserves_value_types(values):
    model = Model(MockOps())
    native_values = model.iterator(values).ref[1]
    assert native_values == tuple(values)
    assert [type(value) for value in native_values] == [type(value) for value in values]


def test_nonlinear_functions():
    ops = MockOps()
    x = Variable(ops, 0)
    expr = sin(x) + 1
    assert expr.ref == ("+", ("sin", ("var", 0)), 1)


@pytest.mark.parametrize(
    ("compare", "constructor"),
    [(operator.le, "LessThan"), (operator.ge, "GreaterThan"), (operator.eq, "EqualTo")],
)
def test_numeric_comparison_preserves_function(compare, constructor):
    ops = MockOps()
    x = Variable(ops, 0)
    for func in (x, x + 1):
        con = compare(func, 10)
        assert con.func is func
        assert con.set == (constructor, 10.0)


@pytest.mark.parametrize(
    ("compare", "constructor"),
    [(operator.le, "LessThan"), (operator.ge, "GreaterThan"), (operator.eq, "EqualTo")],
)
def test_symbolic_comparison_normalizes_into_native_zero_set(compare, constructor):
    ops = MockOps()
    x, y = Variable(ops, 0), Variable(ops, 1)
    con = compare(x + 1, y + 2)
    assert con.set == (constructor, 0.0)
    assert con.func.ref == (
        "-", ("+", ("var", 0), 1), ("+", ("var", 1), 2),
    )


@pytest.mark.parametrize(
    ("compare", "constructor"),
    [(operator.le, "GreaterThan"), (operator.ge, "LessThan"), (operator.eq, "EqualTo")],
)
def test_reflected_numeric_comparison(compare, constructor):
    x = Variable(MockOps(), 0)
    con = compare(10, x)
    assert con.func is x
    assert con.set == (constructor, 10.0)


def test_variable_vector_concrete_indexing():
    ops = MockOps()
    start = ops.add_variables(3)
    x = VariableVector(ops, start, 3, "x")
    assert x[2].index == 2
    assert x[2].ref == ("var", 2)
    assert len(x) == 3
    assert [v.index for v in x] == [0, 1, 2]


def test_variable_vector_symbolic_indexing():
    ops = MockOps()
    x = VariableVector(ops, 0, 10, "x")
    i = Node(ops, ops.iterator([0, 1]))
    # 0-based Python index -> 1-based Julia index
    assert x[i].ref == (
        "getindex",
        ("block", 0, 10),
        ("+", ("iterator", (0, 1)), 1),
    )
    assert type(x[i].ref[-1][-1]) is int


def test_parameter_indexing():
    ops = MockOps()
    p = Parameter(ops, [1.0, 2.0, 3.0], "costs")
    assert p[1] == 2.0
    i = Node(ops, ops.iterator([0, 1]))
    assert p[i].ref == (
        "getindex",
        ("data", (1.0, 2.0, 3.0)),
        ("+", ("iterator", (0, 1)), 1),
    )


def test_templates_use_the_same_native_operations():
    ops = MockOps()
    x = VariableVector(ops, 0, 10, "x")
    i = Node(ops, ops.iterator([0, 1]))
    element = x[i]
    assert (element + 1).ref == ("+", element.ref, 1)
    assert (element**2).ref == ("^", element.ref, 2)
    assert sin(element).ref == ("sin", element.ref)
    # JuMP and GenOpt classify expressions in Julia, not in Python.
    assert not hasattr(element, "linear")
