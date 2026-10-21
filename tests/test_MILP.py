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
    x = m.variables(3, binary=True, name="x")
    assert ops.constraints == [(("getindex", x.ref, k + 1), ("ZeroOne",)) for k in range(3)]


def test_integer_variables():
    ops = MockOps()
    m = Model(ops)
    x = m.variables(3, integer=True, name="x")
    assert ops.constraints == [(("getindex", x.ref, k + 1), ("Integer",)) for k in range(3)]


def test_single_binary_variable_with_bounds():
    ops = MockOps()
    m = Model(ops)
    x = m.variable(lower=0, binary=True, name="x")
    assert ops.constraints == [
        (x.ref, ("GreaterThan", 0.0)),
        (x.ref, ("ZeroOne",)),
    ]


def test_binary_takes_precedence_over_integer():
    ops = MockOps()
    m = Model(ops)
    x = m.variable(binary=True, integer=True, name="x")
    assert ops.constraints == [(x.ref, ("ZeroOne",))]


def test_default_is_continuous():
    ops = MockOps()
    m = Model(ops)
    x = m.variables(3, lower=0, name="x")
    assert ops.constraints == [(("getindex", x.ref, k + 1), ("GreaterThan", 0.0)) for k in range(3)]


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
    x = model.variables(3, **kwargs)
    assert len(created) == 1
    assert ops.constraints == [(("getindex", x.ref, k + 1), created[0]) for k in range(3)]


def test_bounds_use_each_arrays_local_variables():
    ops = MockOps()
    model = Model(ops)
    x = model.variables(2, lower=1)
    y = model.variables(2, upper=4)
    assert ops.constraints == [
        (("getindex", x.ref, k), ("GreaterThan", 1.0)) for k in (1, 2)
    ] + [
        (("getindex", y.ref, k), ("LessThan", 4.0)) for k in (1, 2)
    ]
