"""Constraint references and scalar duals, with mock and real backend checks."""

import importlib.util
import math
import os
from types import SimpleNamespace

import pytest

from jumpy import ConstraintRef, backend, bridge_juliacall
from jumpy._highs_moi import NativeSet
from jumpy.model import Model
from mock_ops import MockOps


@pytest.fixture
def native_backend():
    choice = os.environ.get("JUMPY_BACKEND", "juliac")
    if choice == "juliacall":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("the juliacall backend is not available")
        import jumpy.juliacall as jp
    else:
        if backend.find_lib() is None:
            pytest.skip("the juliac backend is not available")
        import jumpy.highs as jp
    return jp


@pytest.mark.parametrize("form", ["comparison", "explicit", "group"])
def test_constraint_returns_native_reference(form):
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    if form == "explicit":
        row = model.constraint(x, ops.MOI.LessThan(1.0))
    else:
        method = model.constraint_group if form == "group" else model.constraint
        row = method(x <= 1)
    assert isinstance(row, ConstraintRef)
    assert row._ops is ops
    assert row.ref == ("constraint", 1)
    assert model.dual(row) == -3.0
    assert ops.dual_calls == [row.ref]


def test_dual_rejects_invalid_foreign_and_closed_references():
    model, other = Model(MockOps()), Model(MockOps())
    x = model.variable()
    for value in (x, x <= 1, object()):
        with pytest.raises(TypeError, match="constraint reference"):
            model.dual(value)
    with pytest.raises(ValueError, match="different model"):
        model.dual(other.constraint(other.variable() <= 1))
    row = model.constraint(x <= 1)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.dual(row)


def test_compiled_constraint_and_dual_pointer_forwarding():
    calls = []
    lib = SimpleNamespace(
        jumpy_new_model=lambda: 99,
        jumpy_free_model=lambda model: 0,
        jumpy_free_set=lambda set_id: 0,
        jumpy_add_constraint=lambda model, func, set_id: 456,
        jumpy_dual=lambda model, row: calls.append((model, row)) or -2.0,
    )
    ops = backend.JuliacOps(lib)
    set_ = NativeSet(lib, 123)
    try:
        assert ops.add_constraint(321, set_) == 456
        assert ops.dual(456) == -2.0
        assert calls == [(99, 456)]
        lib.jumpy_dual = lambda *_: math.nan
        with pytest.raises(RuntimeError, match="scalar constraint dual"):
            ops.dual(456)
        lib.jumpy_add_constraint = lambda *_: None
        with pytest.raises(RuntimeError, match="add constraint"):
            ops.add_constraint(321, set_)
    finally:
        set_.close()
        ops.free()


def test_juliacall_returns_and_queries_actual_constraint():
    calls = []
    ops = bridge_juliacall.JuliaCallOps.__new__(bridge_juliacall.JuliaCallOps)
    ops._model = object()
    ops._jl = SimpleNamespace(MOI=SimpleNamespace(AbstractScalarSet=object), isa=lambda *_: True)
    native_ref = object()
    ops._jump = SimpleNamespace(
        add_constraint=lambda model, func, set_: native_ref,
        dual=lambda model, row: calls.append((model, row)) or -2.0,
    )
    assert ops.add_constraint("expression", "set") is native_ref
    assert ops.dual(native_ref) == -2.0
    assert calls == [(ops._model, native_ref)]


@pytest.mark.parametrize("sense,constructor,expected", [
    ("minimize", "GreaterThan", 1.0),
    ("maximize", "LessThan", -1.0),
    ("minimize", "EqualTo", 1.0),
    ("maximize", "EqualTo", -1.0),
])
def test_native_scalar_dual_signs_and_dirty_results(native_backend, sense, constructor, expected):
    jp = native_backend
    model = jp.Model()
    try:
        x = model.variable()
        row = model.constraint(x + 0, getattr(jp.MOI, constructor)(2.0))
        model.objective = getattr(jp, sense)(x)
        with pytest.raises(Exception):
            model.dual(row)
        model.optimize()
        assert model.has_duals()
        assert model.dual(row) == pytest.approx(expected)
        model.constraint(x + 0 <= 3)
        if os.environ.get("JUMPY_BACKEND", "juliac") == "juliacall":
            assert not model.has_duals()
            with pytest.raises(Exception):
                model.dual(row)
        else:
            # Direct models leave result availability to HiGHS.
            assert model.has_duals()
            assert model.dual(row) == pytest.approx(expected)
    finally:
        model.close()


def test_native_group_and_foreign_references_are_rejected(native_backend):
    jp = native_backend
    model, other = jp.Model(), jp.Model()
    try:
        x = model.variables(2, lower=0, upper=4)
        row = model.constraint(x[0] + 0 <= 2)
        i = model.iterator(range(2))
        group = model.constraint_group(x[i] + 0 <= 3)
        model.objective = jp.maximize(x[0] + x[1])
        model.optimize()
        assert model.dual(row) == pytest.approx(-1)
        with pytest.raises(Exception):
            model.dual(group)
        with pytest.raises(ValueError, match="different model"):
            other.dual(row)
    finally:
        model.close()
        other.close()


def test_native_mip_has_no_scalar_dual(native_backend):
    jp = native_backend
    model = jp.Model()
    try:
        x = model.variable(binary=True)
        row = model.constraint(x + 0 <= 1)
        model.objective = jp.maximize(x)
        model.optimize()
        assert not model.has_duals()
        with pytest.raises(Exception):
            model.dual(row)
    finally:
        model.close()
