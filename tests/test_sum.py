"""Lazy summation retains a native template instead of expanding its terms."""

import importlib.util
import os
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


def test_lazy_sum_forwards_one_template_once():
    ops = MockOps()
    model = Model(ops)
    x = model.variables(3)
    cost = model.parameter([2, 3, 4])
    i = model.iterator(range(3))
    template = cost[i] * x[i]
    calls = []
    ops.apply = lambda op, args: calls.append((op, args)) or (op, *args)

    result = model.lazy_sum(template)

    assert result.ref == ("lazy_sum", template.ref)
    assert result._ops is ops
    assert calls == [("lazy_sum", [template.ref])]


def test_lazy_sum_rejects_foreign_invalid_and_closed_inputs():
    ops = MockOps()
    model, other = Model(ops), Model(MockOps())
    foreign = other.variable()
    calls = []
    ops.apply = lambda op, args: calls.append((op, args))

    with pytest.raises(ValueError, match="different models"):
        model.lazy_sum(foreign)
    for invalid in (object(), [], iter(())):
        with pytest.raises(TypeError, match="expression"):
            model.lazy_sum(invalid)
    assert calls == []
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.lazy_sum(foreign)


def test_compiled_lazy_sum_uses_generic_apply_and_checks_null():
    calls = []

    def native(model, op, values, count):
        calls.append((model, op, tuple(values[:count])))
        return lib.result
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 99, jumpy_free_model=lambda _: 0,
        jumpy_apply=native, result=456,
    )
    ops = backend.JuliacOps(lib)
    try:
        assert ops.apply("lazy_sum", [111]) == 456
        assert calls == [(99, b"lazy_sum", (111,))]
        lib.result = 0
        with pytest.raises(RuntimeError, match="Failed"):
            ops.apply("lazy_sum", [111])
    finally:
        ops.free()


def test_juliacall_lazy_sum_uses_shared_generic_apply():
    calls = []
    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._jl = SimpleNamespace(Symbol=lambda name: ("Symbol", name))
    ops._any_vec = lambda *args: list(args)
    ops._jump = SimpleNamespace(
        apply=lambda op, args: calls.append((op, args)) or "lazy sum",
    )
    assert ops.apply("lazy_sum", ["template"]) == "lazy sum"
    assert calls == [(("Symbol", "lazy_sum"), ["template"])]


def _interface():
    if os.environ.get("JUMPY_BACKEND", "juliac") == "juliacall":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("JuliaCall is not installed")
        import jumpy.juliacall as jp
    else:
        if backend.find_lib() is None:
            pytest.skip("The compiled backend is not available")
        import jumpy.highs as jp
    return jp


def _model(jp):
    if os.environ.get("JUMPY_BACKEND", "juliac") == "juliacall":
        jl = bridge_juliacall._julia()
        jl.seval("import HiGHS")
        return jp.Model(jl.HiGHS.Optimizer)
    return jp.Model()


def test_native_lazy_sum_preserves_template_and_solves_lp(monkeypatch):
    jp = _interface()
    model = _model(jp)
    try:
        x = model.variables(3, lower=0)
        cost = model.parameter([2, 3, 4])
        i = model.iterator(range(3))
        template = cost[i] * x[i]
        calls = []
        native_apply = model._ops.apply

        def apply(op, args):
            calls.append((op, args))
            return native_apply(op, args)

        monkeypatch.setattr(model._ops, "apply", apply)
        total = model.lazy_sum(template)
        assert len(calls) == 1
        assert calls[0][0] == "lazy_sum"
        assert len(calls[0][1]) == 1
        assert calls[0][1][0] is template.ref
        if os.environ.get("JUMPY_BACKEND", "juliac") == "juliacall":
            jl = model._ops._jl
            assert jl.isa(total.ref, jl.GenOpt.LazySum)
            func = jl.JuMP.moi_function(total.ref)
            assert jl.isa(func, jl.GenOpt.PolynomialFunction)
            raw = jl.seval("f -> only(f.func.args)")(func)
            assert jl.isa(raw, jl.GenOpt.SumGenerator)
            assert int(jl.length(raw.iterators)) == 1
            assert int(jl.seval("s -> length(only(s.iterators))")(raw)) == 3
        model.constraint_group(x[i] >= i + 1)
        model.constraint(model.lazy_sum(x[i] + 1) <= 12)
        model.objective = jp.minimize(total)
        assert model.optimize() == 1
        assert [model.value(variable) for variable in x] == pytest.approx([1, 2, 3])
        assert model.objective_value() == pytest.approx(20)
    finally:
        model.close()


def test_native_lazy_sum_quadratic_objective():
    jp = _interface()
    model = _model(jp)
    try:
        x = model.variables(3, lower=0)
        i = model.iterator(range(3))
        model.constraint_group(x[i] >= 1)
        model.objective = jp.minimize(model.lazy_sum((x[i] - (i + 1)) ** 2))
        assert model.optimize() == 1
        assert [model.value(variable) for variable in x] == pytest.approx(
            [1, 2, 3], abs=1e-3,
        )
        assert model.objective_value() == pytest.approx(0, abs=1e-6)
    finally:
        model.close()


def test_native_lazy_sum_integer_objective():
    jp = _interface()
    model = _model(jp)
    try:
        x = model.variables(3, binary=True)
        i = model.iterator(range(3))
        model.constraint(model.lazy_sum(x[i]) <= 2)
        model.objective = jp.maximize(model.lazy_sum((i + 1) * x[i]))
        assert model.optimize() == 1
        assert [model.value(variable) for variable in x] == pytest.approx([0, 1, 1])
        assert model.objective_value() == pytest.approx(5)
    finally:
        model.close()


def test_examodels_receives_one_objective_template():
    if os.environ.get("JUMPY_BACKEND", "juliac") != "juliacall":
        pytest.skip("A custom optimizer requires JuliaCall")
    jp = _interface()
    jl = bridge_juliacall._julia()
    if not jl.seval('Base.find_package("ExaModels") !== nothing'):
        pytest.skip("ExaModels is not installed in the JuliaCall environment")
    jl.seval("import ExaModels")
    optimizer = jl.ExaModels.Optimizer(jl.nothing)
    model = jp.Model(lambda: optimizer)
    try:
        x = model.variables(4)
        cost = model.parameter([2, 3, 4, 5])
        i = model.iterator(range(4))
        model.objective = jp.minimize(model.lazy_sum(cost[i] * x[i]))
        jl.seval("m -> MOI.Utilities.attach_optimizer(JuMP.backend(m))")(
            model._ops._model,
        )
        assert int(jl.length(optimizer.objs)) == 1
        assert int(jl.seval("o -> length(only(o.objs).data)")(optimizer)) == 4
    finally:
        model.close()
