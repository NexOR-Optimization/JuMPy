"""
Tests that binary/integer variables produce the right MOI calls:
VariableIndex-in-ZeroOne / VariableIndex-in-Integer constraints, as in MOI.
"""

import pytest

from jumpy.model import Model
from mock_ops import MockOps


def test_binary_variables():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, binary=True, name="x")
    assert ops.constraints == [(("var", k), ("ZeroOne",)) for k in range(3)]


def test_integer_variables():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, integer=True, name="x")
    assert ops.constraints == [(("var", k), ("Integer",)) for k in range(3)]


def test_single_binary_variable_with_bounds():
    ops = MockOps()
    m = Model(ops)
    m.variable(lower=0, binary=True, name="x")
    assert ops.constraints == [
        (("var", 0), ("GreaterThan", 0.0)),
        (("var", 0), ("ZeroOne",)),
    ]


def test_binary_takes_precedence_over_integer():
    ops = MockOps()
    m = Model(ops)
    m.variable(binary=True, integer=True, name="x")
    assert ops.constraints == [(("var", 0), ("ZeroOne",))]


def test_default_is_continuous():
    ops = MockOps()
    m = Model(ops)
    m.variables(3, lower=0, name="x")
    assert ops.constraints == [(("var", k), ("GreaterThan", 0.0)) for k in range(3)]


@pytest.mark.parametrize(
    ("kwargs", "constructor", "args"),
    [
        ({"lower": 1}, "GreaterThan", (1.0,)),
        ({"upper": 2}, "LessThan", (2.0,)),
        ({"binary": True}, "ZeroOne", ()),
        ({"integer": True}, "Integer", ()),
    ],
)
def test_variable_block_constructs_each_set_once(monkeypatch, kwargs, constructor, args):
    ops = MockOps()
    created = []

    def construct(*values):
        assert values == args
        set_ = object()
        created.append(set_)
        return set_

    monkeypatch.setattr(ops.MOI, constructor, construct)
    model = Model(ops)
    model.variables(3, **kwargs)
    assert len(created) == 1
    assert ops.constraints == [(("var", k), created[0]) for k in range(3)]
