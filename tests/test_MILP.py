"""
Tests that binary/integer variables produce the right MOI calls:
VariableIndex-in-ZeroOne / VariableIndex-in-Integer constraints, as in MOI.
"""

from jumpy.model import Model
from mock_ops import MockOps


def test_binary_variables():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, binary=True, name="x")
    assert ops.constraints == [(("var", k), "binary", 0.0) for k in range(3)]


def test_integer_variables():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, integer=True, name="x")
    assert ops.constraints == [(("var", k), "integer", 0.0) for k in range(3)]


def test_single_binary_variable_with_bounds():
    ops = MockOps()
    m = Model(ops)
    m.variable(lower=0, binary=True, name="x")
    assert ops.constraints == [
        (("var", 0), ">=", 0.0),
        (("var", 0), "binary", 0.0),
    ]


def test_binary_takes_precedence_over_integer():
    ops = MockOps()
    m = Model(ops)
    m.variable(binary=True, integer=True, name="x")
    assert ops.constraints == [(("var", 0), "binary", 0.0)]


def test_default_is_continuous():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, lower=0, name="x")
    assert all(sense == ">=" for _, sense, _ in ops.constraints)
