"""Native summation, independently of result/constraint-reference APIs."""

import importlib.util
import math
import os
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


def test_sum_forwards_native_terms_once_and_accepts_empty_iterables():
    ops = MockOps()
    model = Model(ops)
    x, y = model.variables(2)
    terms = [x, 2 * y, 3]
    result = model.sum(term for term in terms)
    assert result.ref == ("sum", x.ref, terms[1].ref, 3)
    assert ops.sum_calls == [[x.ref, terms[1].ref, 3]]
    assert model.sum([]).ref == ("sum",)


def test_sum_rejects_foreign_invalid_and_closed_inputs():
    ops = MockOps()
    model, other = Model(ops), Model(MockOps())
    with pytest.raises(ValueError, match="different models"):
        model.sum([model.variable(), other.variable()])
    with pytest.raises(TypeError, match="expression"):
        model.sum([object()])
    assert ops.sum_calls == []
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.sum([])


@pytest.mark.parametrize("terms", [[], [111, 222]])
def test_compiled_sum_forwards_handles_and_checks_null(terms):
    calls = []
    def native(model, values, count):
        calls.append((model, tuple(values[:count])))
        return lib.result
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 99, jumpy_free_model=lambda _: 0,
        jumpy_sum=native, result=456,
    )
    ops = backend.JuliacOps(lib)
    try:
        assert ops.sum_terms(terms) == 456
        assert calls == [(99, tuple(terms))]
        lib.result = 0
        with pytest.raises(RuntimeError, match="Failed"):
            ops.sum_terms(terms)
    finally:
        ops.free()


def test_juliacall_sum_forwards_shared_helper():
    calls = []
    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._model = object()
    ops._any_vec = lambda *args: list(args)
    ops._jump = SimpleNamespace(
        sum_terms=lambda model, terms: calls.append((model, terms)) or "sum",
    )
    assert ops.sum_terms(["x", "y"]) == "sum"
    assert calls == [(ops._model, ["x", "y"])]


def test_native_sum_expression_values_and_quadratic_objective():
    if os.environ.get("JUMPY_BACKEND", "juliac") == "juliacall":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("JuliaCall is not installed")
        import jumpy.juliacall as jp
    else:
        if backend.find_lib() is None:
            pytest.skip("The compiled backend is not available")
        import jumpy.highs as jp
    model = jp.Model()
    try:
        x, y = model.variables(2, lower=0, upper=5)
        empty, numeric = model.sum([]), model.sum([2, 3.5])
        affine = model.sum([x, 2 * y, 3])
        doubled = model.sum([affine, affine])
        quadratic = model.sum([(x - 2) ** 2, (y - 3) ** 2])
        nonlinear = model.sum([jp.sin(x), y])
        model.objective = jp.minimize(quadratic)
        model.optimize()
        assert model.value(x) == pytest.approx(2, abs=1e-6)
        assert model.value(y) == pytest.approx(3, abs=1e-6)
        for expr, expected in [(empty, 0), (numeric, 5.5), (affine, 11),
                               (doubled, 22), (quadratic, 0),
                               (nonlinear, math.sin(2) + 3)]:
            assert model.value(expr) == pytest.approx(expected, abs=2e-6)
    finally:
        model.close()
