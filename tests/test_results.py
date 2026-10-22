"""Result queries and nonoptimal termination, without initializing Julia."""

import math
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


def test_solution_queries_are_native_and_not_cached():
    ops = MockOps()
    model = Model(ops)
    assert model.termination_status() == 1
    assert model.has_values() is True
    assert model.has_duals() is True
    assert model.objective_value() == 2.5
    ops.status = 12
    ops.primal_available = False
    ops.dual_available = False
    ops.objective_result = 7.0
    assert model.termination_status() == 12
    assert model.has_values() is False
    assert model.has_duals() is False
    assert model.objective_value() == 7.0


def test_nonoptimal_status_does_not_discard_available_values():
    ops = MockOps()
    model = Model(ops)
    assert model.optimize() == 1
    ops.status = 12  # MOI.TIME_LIMIT
    with pytest.raises(RuntimeError, match="termination status 12"):
        model.optimize()
    assert model.optimize(require_optimal=False) == 12
    assert model.has_values()
    assert model.objective_value() == 2.5
    assert ops.optimize_calls == 3


@pytest.mark.parametrize(
    "method", ["optimize", "termination_status", "has_values", "has_duals", "objective_value"],
)
def test_result_operations_reject_closed_models(method):
    model = Model(MockOps())
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        getattr(model, method)()


class ResultLib:
    def __init__(self):
        self.scalar = 2.5
        self.flag = 1
        self.status = 12

    def jumpy_new_model(self):
        return 99

    def jumpy_free_model(self, model):
        return 0

    def jumpy_has_values(self, model):
        return self.flag

    def jumpy_has_duals(self, model):
        return self.flag

    def jumpy_objective_value(self, model):
        return self.scalar

    def jumpy_termination_status(self, model):
        return self.status

    def jumpy_optimize(self, model):
        return self.status


def test_compiled_result_queries_forward_native_values():
    lib = ResultLib()
    ops = backend.JuliacOps(lib)
    try:
        assert ops.objective_value() == 2.5
        assert ops.has_values() is True
        assert ops.has_duals() is True
        assert ops.termination_status() == ops.optimize() == 12
        lib.flag = 0
        assert ops.has_values() is ops.has_duals() is False
    finally:
        ops.free()


@pytest.mark.parametrize(
    "method,field,value",
    [
        ("objective_value", "scalar", math.nan),
        ("has_values", "flag", -1), ("has_duals", "flag", -1),
        ("termination_status", "status", -1), ("optimize", "status", -1),
    ],
)
def test_compiled_errors_are_not_treated_as_results(method, field, value):
    lib = ResultLib()
    setattr(lib, field, value)
    ops = backend.JuliacOps(lib)
    try:
        with pytest.raises(RuntimeError, match="Failed"):
            getattr(ops, method)()
    finally:
        ops.free()


def test_juliacall_result_operations_forward_native_model():
    calls = []

    def native(name, value):
        def call(model):
            calls.append((name, model))
            return value
        return call

    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._model = object()
    ops._jl = SimpleNamespace(Int=int)
    ops._jump = SimpleNamespace(**{
        name: native(name, value) for name, value in {
            "optimize": None, "termination_status": 12, "has_values": True,
            "has_duals": False, "objective_value": 3.5,
        }.items()
    })
    assert ops.optimize() == ops.termination_status() == 12
    assert ops.has_values() is True
    assert ops.has_duals() is False
    assert ops.objective_value() == 3.5
    assert calls == [
        (name, ops._model) for name in (
            "optimize", "termination_status", "termination_status", "has_values",
            "has_duals", "objective_value",
        )
    ]
