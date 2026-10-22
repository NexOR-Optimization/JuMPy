"""Solve-time limits: validation, both transports, and native set/clear."""

import importlib.util
import math
import os
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


class TimeLimitOps(MockOps):
    def __init__(self):
        super().__init__()
        self.time_limits = []

    def set_time_limit_sec(self, seconds):
        self.time_limits.append(seconds)


def test_time_limit_sets_and_clears_native_option():
    ops = TimeLimitOps()
    model = Model(ops)
    model.set_time_limit_sec(5)
    model.set_time_limit_sec(None)
    model.set_time_limit_sec(3)
    model.unset_time_limit_sec()
    assert ops.time_limits == [5.0, None, 3.0, None]


@pytest.mark.parametrize("seconds", [0, -1, math.nan, math.inf, -math.inf])
def test_invalid_time_limit_is_rejected_before_native_call(seconds):
    ops = TimeLimitOps()
    with pytest.raises(ValueError, match="positive and finite"):
        Model(ops).set_time_limit_sec(seconds)
    assert ops.time_limits == []


def test_time_limit_rejects_closed_model():
    model = Model(TimeLimitOps())
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.set_time_limit_sec(1)
    with pytest.raises(RuntimeError, match="closed"):
        model.unset_time_limit_sec()


def test_compiled_time_limit_encodes_none_as_nan_and_checks_errors():
    calls = []
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 99,
        jumpy_free_model=lambda model: 0,
        jumpy_set_time_limit_sec=lambda *args: calls.append(args) or 0,
    )
    ops = backend.JuliacOps(lib)
    try:
        ops.set_time_limit_sec(4.0)
        ops.set_time_limit_sec(None)
        assert calls[0] == (99, 4.0)
        assert calls[1][0] == 99 and math.isnan(calls[1][1])
        lib.jumpy_set_time_limit_sec = lambda *args: -1
        with pytest.raises(RuntimeError, match="Failed to set time limit"):
            ops.set_time_limit_sec(1.0)
    finally:
        ops.free()


def test_juliacall_time_limit_encodes_none_as_nan():
    calls = []
    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._model = object()
    ops._jump = SimpleNamespace(set_time_limit_sec=lambda *args: calls.append(args))
    ops.set_time_limit_sec(4.0)
    ops.set_time_limit_sec(None)
    assert calls[0] == (ops._model, 4.0)
    assert calls[1][0] is ops._model and math.isnan(calls[1][1])


def test_time_limit_can_be_set_cleared_and_solved():
    choice = os.environ.get("JUMPY_BACKEND", "juliac")
    if choice == "juliacall":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("the juliacall backend is not available")
        import jumpy.juliacall as jp
    else:
        if backend.find_lib() is None:
            pytest.skip("the juliac backend is not available")
        import jumpy.highs as jp
    model = jp.Model()
    try:
        x = model.variable(lower=1)
        model.objective = jp.minimize(x)
        model.set_time_limit_sec(10)
        if choice == "juliacall":
            assert model._ops._jl.JuMP.time_limit_sec(model._ops._model) == 10.0
        model.optimize()
        assert model.value(x) == pytest.approx(1)
        model.unset_time_limit_sec()
        if choice == "juliacall":
            assert model._ops._jl.JuMP.time_limit_sec(model._ops._model) is None
        model.optimize()
        assert model.value(x) == pytest.approx(1)
    finally:
        model.close()
