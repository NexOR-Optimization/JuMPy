"""
End-to-end tests that solve models with HiGHS.

Backend is selected with the JUMPY_BACKEND environment variable:
    JUMPY_BACKEND=juliac   (default) requires the compiled library
        (julia/README.md), no Julia needed
    JUMPY_BACKEND=juliacall requires Julia + juliacall:
        pip install jumpy[juliacall]

The whole module is skipped when the selected backend is not available.
"""

import importlib.util
import operator
import os
import subprocess
import sys

import pytest

from jumpy import backend, minimize, maximize

BACKEND = os.environ.get("JUMPY_BACKEND", "juliac")
if BACKEND == "juliacall":
    import jumpy.juliacall as jp
else:
    import jumpy.highs as jp


def _backend_available():
    if BACKEND == "juliacall":
        return importlib.util.find_spec("juliacall") is not None
    # A set JUMPY_LIB counts as available even if the file is missing:
    # loading must then fail loudly instead of the suite silently skipping.
    return backend.find_lib() is not None


pytestmark = pytest.mark.skipif(
    not _backend_available(), reason=f"the {BACKEND} backend is not available"
)


def _model():
    return jp.Model()


def test_simple_lp():
    """min x + y  s.t.  x + y >= 10, x >= 0, y >= 0"""
    m = _model()
    x = m.variable(lower=0, name="x")
    y = m.variable(lower=0, name="y")

    m.constraint(x + y >= 10)
    m.objective = minimize(x + y)
    m.optimize()

    assert abs(m.value(x) + m.value(y) - 10.0) < 1e-6


@pytest.mark.parametrize("method", ["constraint", "constraint_group"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    ("constructor", "compare", "objective"),
    [
        ("LessThan", operator.le, maximize),
        ("GreaterThan", operator.ge, minimize),
        ("EqualTo", operator.eq, minimize),
    ],
)
def test_constraint_group_native_sets(method, explicit, constructor, compare, objective):
    """All three native sets vectorize correctly, including nonzero bounds."""
    m = _model()
    try:
        x = m.variables(10, lower=0, upper=3, name="x")
        i = m.iterator(range(10))
        con = (
            jp.Constraint(x[i] + 0, getattr(jp.MOI, constructor)(2.0))
            if explicit else compare(x[i], 2.0)
        )
        getattr(m, method)(con)
        m.objective = objective(sum(x))
        m.optimize()
        assert [m.value(v) for v in x] == pytest.approx([2.0] * len(x))
    finally:
        m.close()


def test_constraint_group_consecutive():
    """
    min x[0]+x[1]+x[2]  s.t.  x[i] + x[i+1] >= 2 for i in 0..8, x[i] >= 0
    """
    m = _model()
    x = m.variables(10, lower=0, name="x")

    i = m.iterator(range(9))
    m.constraint_group(x[i] + x[i + 1] >= 2)

    m.objective = minimize(x[0] + x[1] + x[2])
    m.optimize()

    for k in range(9):
        assert m.value(x[k]) + m.value(x[k + 1]) >= 2.0 - 1e-6


def test_parameter_in_constraint_group():
    """
    min sum(x)  s.t.  x[i] >= demand[i], x[i] >= 0
    demand = [1, 2, 3, 4, 5]
    Optimal: x[i] = demand[i], obj = 15
    """
    m = _model()
    x = m.variables(5, lower=0, name="x")
    demand = m.parameter([1.0, 2.0, 3.0, 4.0, 5.0], name="demand")

    i = m.iterator(range(5))
    m.constraint_group(x[i] >= demand[i])

    m.objective = minimize(sum(x))
    m.optimize()

    total = sum(m.value(v) for v in x)
    assert abs(total - 15.0) < 1e-6


def test_multidim_constraint_group():
    """
    2D indexing: x[3*i + j] >= 1 for i in 0..2, j in 0..2
    9 variables, all >= 1
    """
    m = _model()
    x = m.variables(9, lower=0, name="x")

    i = m.iterator(range(3))
    j = m.iterator(range(3))
    m.constraint_group(x[3 * i + j] >= 1)

    m.objective = minimize(sum(x))
    m.optimize()

    total = sum(m.value(v) for v in x)
    assert abs(total - 9.0) < 1e-6


def test_constraint_group_over_bounded_variables():
    """
    x[i] >= 0 as a group on variables that already have lower=0: must be
    rows (like JuMP's @constraint), not clashing variable bounds.
    """
    m = _model()
    x = m.variables(4, lower=0, name="x")

    i = m.iterator(range(4))
    m.constraint_group(x[i] >= 0)

    m.objective = minimize(sum(x))
    m.optimize()

    assert abs(sum(m.value(v) for v in x)) < 1e-6


def test_maximize():
    """max x  s.t.  x <= 42, x >= 0"""
    m = _model()
    x = m.variable(lower=0, upper=42, name="x")

    m.objective = maximize(x)
    m.optimize()

    assert abs(m.value(x) - 42.0) < 1e-6

def test_knapsack():
    m = _model()
    x = m.variables(3, binary=True, name="x")
    val = [60, 100, 120]
    wt  = [10, 20, 30]
    m.constraint(sum(wt[i]*x[i] for i in range(3)) <= 50)
    m.objective = maximize(sum(val[i]*x[i] for i in range(3)))
    m.optimize()
    
    assert abs(sum(val[i] * m.value(x[i]) for i in range(3)) - 220.0) < 1e-6

def test_assignement():
    cost = [
        [9, 2, 7],
        [6, 4, 3],
        [5, 8, 1],
    ]
    m = _model()
    x = m.variables(9, binary=True, name="x")
    def xij(i, j): return x[3*i + j]
    for i in range(3):
        m.constraint(sum(xij(i, j) for j in range(3)) == 1)
    for j in range(3):
        m.constraint(sum(xij(i, j) for i in range(3)) == 1)
    m.objective = minimize(sum(cost[i][j]*xij(i, j)
                                  for i in range(3) for j in range(3)))
    m.optimize()

    assert abs(sum(cost[i][j] * m.value(x[i * 3 + j]) for i in range(3) for j in range(3)) - 9.0) < 1e-6

def test_integer_production_planning():
    m = _model()
    p = m.variable(lower=0, integer=True, name="p")
    q = m.variable(lower=0, integer=True, name="q")
    m.constraint(p <= 4)
    m.constraint(2*q <= 12)
    m.constraint(3*p + 2*q <= 18)
    m.objective = maximize(3*p + 5*q)
    m.optimize()

    assert abs(3 * m.value(p) + 5 * m.value(q) - 36.0) < 1e-6

def test_set_cover():
    m = _model()
    s = m.variables(4, binary=True, name="s")
    sets = {0: [0, 3], 1: [0, 1], 2: [1, 2], 3: [2, 3]}
    for elem in range(4):
        m.constraint(sum(s[k] for k in sets[elem]) >= 1)
    m.objective = minimize(sum(s[k] for k in range(4)))
    m.optimize()
    assert abs(sum(m.value(s[k]) for k in range(4)) - 2.0) < 1e-6


def test_native_less_than_constraint():
    m = _model()
    try:
        x, y = m.variables(2)
        m.constraint(x + y, jp.MOI.LessThan(1.0))
        m.objective = jp.maximize(x + y)
        m.optimize()
        assert m.value(x) + m.value(y) == pytest.approx(1.0)
    finally:
        m.close()


@pytest.mark.parametrize(
    ("constructor", "rhs", "direction"),
    [("LessThan", 7.0, "max"), ("GreaterThan", 7.0, "min"), ("EqualTo", 7.0, "min")],
)
def test_native_set_normalizes_expression_constant(constructor, rhs, direction):
    m = _model()
    try:
        x = m.variable(lower=0)
        m.constraint(2 * x + 3, getattr(jp.MOI, constructor)(rhs))
        m.objective = (jp.maximize if direction == "max" else jp.minimize)(x)
        m.optimize()
        assert m.value(x) == pytest.approx(2.0)
    finally:
        m.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_constraint_normalization_does_not_mutate_reused_expression(explicit):
    model = jp.Model()
    try:
        x = model.variable(lower=0)
        expr = x + 3
        for rhs in (5.0, 4.0):
            if explicit:
                model.constraint(expr, jp.MOI.LessThan(rhs))
            else:
                model.constraint(expr <= rhs)
        model.objective = jp.maximize(expr)
        model.optimize()
        assert model.value(x) == pytest.approx(1.0)
    finally:
        model.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_native_bare_variable_bound_and_expression_row(explicit):
    m = _model()
    try:
        x = m.variable()
        # An expression simplifying to x must remain a row, not try to add a
        # second VariableIndex-in-GreaterThan bound on the same variable.
        if explicit:
            m.constraint(x, jp.MOI.GreaterThan(0.0))
            m.constraint(x + 0, jp.MOI.GreaterThan(2.0))
            m.constraint(x, jp.MOI.LessThan(4.0))
        else:
            m.constraint(x >= 0.0)
            m.constraint(x + 0 >= 2.0)
            m.constraint(x <= 4.0)
        m.objective = jp.minimize(x)
        m.optimize()
        assert m.value(x) == pytest.approx(2.0)
    finally:
        m.close()


@pytest.mark.parametrize(
    ("constructor", "bound", "optimum"),
    [("ZeroOne", 1.5, 1.0), ("Integer", 2.5, 2.0)],
)
def test_native_integrality_set(constructor, bound, optimum):
    m = _model()
    try:
        x = m.variable(lower=0)
        m.constraint(x, getattr(jp.MOI, constructor)())
        m.constraint(x + 0, jp.MOI.LessThan(bound))
        m.objective = jp.maximize(x)
        m.optimize()
        assert m.value(x) == pytest.approx(optimum)
    finally:
        m.close()


@pytest.mark.parametrize("value", [0, 0.0, 2, 2.0])
def test_native_numeric_constraint(value):
    m = _model()
    try:
        x = m.variable(lower=0, upper=1)
        m.constraint(value, jp.MOI.LessThan(3.0))
        m.objective = jp.maximize(x)
        m.optimize()
        assert m.value(x) == pytest.approx(1.0)
    finally:
        m.close()


def test_native_quadratic_objective():
    """Ordinary arithmetic must create a JuMP quadratic, not a nonlinear tree."""
    model = jp.Model()
    try:
        x = model.variable(lower=0, upper=5)
        model.objective = jp.minimize((x - 2) ** 2)
        model.optimize()
        assert model.value(x) == pytest.approx(2.0, abs=1e-6)
    finally:
        model.close()


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("kind", ["quadratic", "nonlinear"])
def test_unsupported_constraint_reports_native_error(group, kind):
    """Unsupported expressions must error, never be silently treated as affine."""
    if BACKEND != "juliacall":
        # Julia caches its stderr at startup, before pytest's per-test capture.
        # Start fresh to capture the actual native unsupported diagnostic.
        script = f"""
import jumpy.highs as jp
model = jp.Model()
variables = model.variables(2, lower=0, upper=5)
x = variables[model.iterator(range(2))] if {group!r} else variables[0]
func = x**2 if {kind!r} == "quadratic" else jp.sin(x)
try:
    model.constraint(func, jp.MOI.LessThan(1.0))
except RuntimeError as failure:
    assert "constraint" in str(failure)
else:
    raise AssertionError("HiGHS unexpectedly accepted the constraint")
finally:
    model.close()
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        detail = result.stderr
        assert any(text in detail.lower() for text in (
            "unsupported", "not supported", "does not support",
        )), detail
        return
    model = jp.Model()
    try:
        variables = model.variables(2, lower=0, upper=5)
        x = variables[model.iterator(range(2))] if group else variables[0]
        func = x**2 if kind == "quadratic" else jp.sin(x)
        with pytest.raises(Exception) as failure:
            model.constraint(func, jp.MOI.LessThan(1.0))
        detail = str(failure.value)
        assert any(text in detail.lower() for text in (
            "unsupported", "not supported", "does not support",
        )), detail
    finally:
        model.close()


def test_native_set_reuse_across_models():
    set_ = jp.MOI.LessThan(2.0)
    try:
        for _ in range(2):
            m = _model()
            try:
                x = m.variable(lower=0)
                m.constraint(x + 0, set_)
                m.objective = jp.maximize(x)
                m.optimize()
                assert m.value(x) == pytest.approx(2.0)
            finally:
                m.close()
    finally:
        if BACKEND != "juliacall":
            set_.close()


def test_native_constraint_rejects_foreign_expression():
    first, second = _model(), _model()
    try:
        x = first.variable()
        with pytest.raises(ValueError, match="different model"):
            second.constraint(x, jp.MOI.LessThan(1.0))
        with pytest.raises(ValueError, match="different model"):
            second.constraint(x <= 1.0)
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(
    BACKEND == "juliacall", reason="Only compiled set handles are explicitly closed",
)
def test_compiled_constraint_retains_set_after_handle_is_closed():
    m = _model()
    try:
        x = m.variable(lower=0)
        with jp.MOI.LessThan(3.0) as set_:
            m.constraint(x + 0, set_)
        with pytest.raises(RuntimeError, match="closed"):
            m.constraint(x + 0, set_)
        m.objective = jp.maximize(x)
        m.optimize()
        assert m.value(x) == pytest.approx(3.0)
    finally:
        m.close()
