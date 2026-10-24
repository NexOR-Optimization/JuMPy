"""JuliaCall constructors follow JuMP's optimizer and direct-mode semantics."""

import importlib.util
import os

import pytest

import jumpy.juliacall as jp
from jumpy import bridge_juliacall

pytestmark = pytest.mark.skipif(
    os.environ.get("JUMPY_BACKEND", "juliac") != "juliacall"
    or importlib.util.find_spec("juliacall") is None,
    reason="JuliaCall constructor integration tests require the JuliaCall backend",
)


@pytest.fixture
def jl():
    return bridge_juliacall._julia()


@pytest.fixture
def highs(jl):
    jl.seval("import HiGHS")
    return jl.HiGHS


def _solve(model):
    x = model.variable(lower=2)
    model.objective = jp.minimize(1.0 * x)
    assert model.optimize() == 1
    assert model.value(x) == pytest.approx(2)


def test_model_without_optimizer_builds_but_cannot_solve(jl):
    from juliacall import JuliaError

    model = jp.Model()
    try:
        x = model.variable(lower=0)
        model.constraint(x <= 3)
        model.objective = jp.maximize(x)
        assert bool(jl.seval("m -> JuMP.mode(m) == JuMP.AUTOMATIC")(model._ops._model))
        assert bool(jl.seval(
            "m -> JuMP.backend(m).state == MOI.Utilities.NO_OPTIMIZER"
        )(model._ops._model))
        with pytest.raises(JuliaError) as error:
            model.optimize()
        assert bool(jl.isa(error.value.exception, jl.JuMP.NoOptimizer))
    finally:
        model.close()


@pytest.mark.parametrize("add_bridges", [True, False])
def test_native_factory_preserves_jump_options_and_output_defaults(jl, highs, add_bridges):
    model = jp.Model(highs.Optimizer, add_bridges=add_bridges)
    try:
        native = model._ops._model
        has_bridges = jl.seval(
            "m -> JuMP.backend(m).optimizer isa MOI.Bridges.LazyBridgeOptimizer"
        )
        assert bool(has_bridges(native)) is add_bridges
        assert not bool(jl.seval("m -> MOI.get(JuMP.backend(m), MOI.Silent())")(native))
        _solve(model)
    finally:
        model.close()


def test_python_factory_creates_a_distinct_optimizer_for_each_model(jl, highs):
    created = []

    def factory():
        optimizer = highs.Optimizer()
        created.append(optimizer)
        return optimizer

    first, second = jp.Model(factory), jp.Model(factory)
    try:
        assert len(created) == 2
        distinct = jl.seval("(a, b) -> a !== b")
        assert bool(distinct(*created))
        _solve(first)
        _solve(second)
        assert len(created) == 2
    finally:
        first.close()
        second.close()


def test_python_factory_can_compose_optimizer_layers(jl, highs):
    storage = jl.seval("() -> MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())")
    model = jp.Model(lambda: jp.MOI.Utilities.CachingOptimizer(storage(), highs.Optimizer()))
    try:
        _solve(model)
    finally:
        model.close()


def test_factory_validation_is_left_to_jump(jl, highs):
    from juliacall import JuliaError

    with pytest.raises(JuliaError, match="optimizer_constructor"):
        jp.Model(highs.Optimizer())
    with pytest.raises(JuliaError, match="returned an object of type Nothing"):
        jp.Model(lambda: None)


def test_direct_model_uses_the_supplied_optimizer_without_layers(jl, highs):
    optimizer = highs.Optimizer()
    model = jp.direct_model(optimizer)
    try:
        native = model._ops._model
        assert bool(jl.seval("m -> JuMP.mode(m) == JuMP.DIRECT")(native))
        assert bool(jl.seval("(m, opt) -> JuMP.backend(m) === opt")(native, optimizer))
        assert not bool(jp.MOI.get(optimizer, jp.MOI.Silent()))
        _solve(model)
    finally:
        model.close()


def test_direct_highs_does_not_add_genopt_bridges(highs):
    from juliacall import JuliaError

    model = jp.direct_model(highs.Optimizer())
    try:
        x = model.variables(2)
        i = model.iterator(range(2))
        with pytest.raises(JuliaError, match="(?i)unsupported|not support"):
            model.constraint(x[i] <= 1)
    finally:
        model.close()
