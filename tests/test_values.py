"""Value queries use native expression references, never Python column caches."""

import pytest

from jumpy.model import Model
from mock_ops import MockOps


def test_value_queries_the_backend_each_time():
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    expr = 2 * x + 1
    model.optimize()
    assert ops.optimize_calls == 1
    assert ops.value_calls == []
    assert model.value(expr) == 1.25
    ops.value_result = 3.5
    assert model.value(expr) == 3.5
    assert ops.value_calls == [expr.ref, expr.ref]


def test_value_rejects_foreign_expression_before_native_call():
    ops = MockOps()
    model = Model(ops)
    other = Model(MockOps())
    with pytest.raises(ValueError, match="different models"):
        model.value(other.variable())
    assert ops.value_calls == []


def test_value_rejects_closed_model_before_native_call():
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.value(x)
    assert ops.value_calls == []


def test_value_propagates_native_unsolved_error(monkeypatch):
    ops = MockOps()
    model = Model(ops)
    x = model.variable()

    def unsolved(expr):
        assert expr is x.ref
        raise RuntimeError("The native model has no solution")

    monkeypatch.setattr(ops, "value", unsolved)
    with pytest.raises(RuntimeError, match="native model has no solution"):
        model.value(x)
