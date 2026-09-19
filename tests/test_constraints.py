"""Constraint API tests using mock ops, without Julia or a compiled backend."""

import pytest

import jumpy as jp
from jumpy.expressions import Variable
from mock_ops import MockOps


@pytest.fixture
def model():
    return jp.Model(backend=MockOps())


@pytest.mark.parametrize(
    "set_, sense, rhs",
    [
        (jp.LessThan(11), "<=", 11.0),
        (jp.GreaterThan(11), ">=", 11.0),
        (jp.EqualTo(11), "==", 11.0),
    ],
)
def test_function_in_scalar_set(model, set_, sense, rhs):
    x, y = model.variables(2)
    func = x + y + 3

    model.constraint(func, set_)

    assert model._ops.constraints == [(func.moi, sense, rhs)]


@pytest.mark.parametrize(
    "set_, sense, rhs",
    [
        (jp.LessThan(2), "<=", 2.0),
        (jp.GreaterThan(-2), ">=", -2.0),
        (jp.EqualTo(1), "==", 1.0),
        (jp.ZeroOne(), "binary", 0.0),
        (jp.Integer(), "integer", 0.0),
    ],
)
def test_variable_in_set_preserves_variable_function(model, set_, sense, rhs):
    x = model.variable()

    model.constraint(x, set_)

    assert model._ops.constraints == [(("var", 0), sense, rhs)]


def test_comparison_form_is_unchanged(model):
    x = model.variable()

    model.constraint(x + 2 <= 4)

    assert model._ops.constraints == [
        (("-", ("+", ("var", 0), 2.0), 4.0), "<=", 0.0)
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        lambda x: (x,),
        lambda x: (x <= 1, jp.LessThan(1)),
        lambda x: (1, jp.LessThan(1)),
        lambda x: ([x], jp.LessThan(1)),
        lambda x: (object(), jp.LessThan(1)),
        lambda x: (x, object()),
    ],
    ids=["missing-set", "comparison-with-set", "number", "list", "object", "unknown-set"],
)
def test_invalid_arguments_do_not_reach_backend(model, arguments):
    x = model.variable()

    with pytest.raises(TypeError):
        model.constraint(*arguments(x))

    assert model._ops.constraints == []


def test_custom_set_subclass_is_not_treated_as_builtin(model):
    class CustomLessThan(jp.LessThan):
        pass

    x = model.variable()
    with pytest.raises(TypeError, match="Unsupported constraint set"):
        model.constraint(x, CustomLessThan(1))

    assert model._ops.constraints == []


@pytest.mark.parametrize("comparison", [False, True])
def test_foreign_model_constraint_is_rejected(model, comparison):
    x = Variable(MockOps(), 0)
    arguments = (x + 1 <= 2,) if comparison else (x + 1, jp.LessThan(2))

    with pytest.raises(ValueError):
        model.constraint(*arguments)

    assert model._ops.constraints == []
