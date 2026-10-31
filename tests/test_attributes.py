"""Optimizer attributes, as JuMP.set_attribute, in both backends."""

import importlib.util
import os
from types import SimpleNamespace

import pytest

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


def test_set_attribute_forwards_name_and_value():
    ops = MockOps()
    model = Model(ops)
    for name, value in (("solver", "HiGHS"), ("silent", False), ("threads", 2), ("tol", 1e-6)):
        model.set_attribute(name, value)
        assert ops.attributes[name] is value


@pytest.mark.parametrize(("name", "value"), [(1, "x"), ("x", None), ("x", [1])])
def test_set_attribute_rejects_other_types(name, value):
    ops = MockOps()
    with pytest.raises(TypeError):
        Model(ops).set_attribute(name, value)
    assert ops.attributes == {}


def test_set_attribute_rejects_closed_model():
    model = Model(MockOps())
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.set_attribute("solver", "HiGHS")


def test_juliacall_set_attribute_uses_shared_julia_source():
    calls = []
    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._model = object()
    ops._jump = SimpleNamespace(set_attribute=lambda *args: calls.append(args))
    ops.set_attribute("presolve", "off")
    assert calls == [(ops._model, "presolve", "off")]


def test_set_attribute_reaches_the_solver():
    choice = os.environ.get("JUMPY_BACKEND", "juliac")
    if choice == "juliacall":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("the juliacall backend is not available")
        import jumpy.juliacall as jp
        jl = bridge_juliacall._julia()
        jl.seval("import HiGHS")
        model = jp.Model(jl.HiGHS.Optimizer)
    elif choice == "juliac":
        if backend.find_lib("highs") is None:
            pytest.skip("the juliac backend is not available")
        import jumpy.highs as jp
        model = jp.Model()
    else:
        pytest.skip(f"HiGHS options do not apply to JUMPY_BACKEND={choice}")
    try:
        x = model.variable(lower=1)
        model.objective = jp.minimize(x)
        for name, value in (
            ("presolve", "off"),
            ("mip_detect_symmetry", False),
            ("random_seed", 3),
            ("mip_rel_gap", 1e-4),
        ):
            model.set_attribute(name, value)
        model.optimize()
        assert model.value(x) == pytest.approx(1)
        with pytest.raises(Exception):
            model.set_attribute("not_an_option", 1)
    finally:
        model.close()
