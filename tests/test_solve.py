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
import os

import pytest

from jumpy import Model, backend, minimize, maximize

BACKEND = os.environ.get("JUMPY_BACKEND", "juliac")


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
    return Model(backend=BACKEND)


def test_simple_lp():
    """min x + y  s.t.  x + y >= 10, x >= 0, y >= 0"""
    m = _model()
    x = m.variable(lower=0, name="x")
    y = m.variable(lower=0, name="y")

    m.constraint(x + y >= 10)
    m.objective = minimize(x + y)
    m.optimize()

    assert abs(m.value(x) + m.value(y) - 10.0) < 1e-6


def test_constraint_group_lp():
    """
    min sum(x)  s.t.  x[i] >= 1 for i in 0..9, x[i] >= 0
    Optimal: all x[i] = 1, obj = 10
    """
    m = _model()
    x = m.variables(10, lower=0, name="x")

    i = m.iterator(range(10))
    m.constraint_group(x[i] >= 1)

    m.objective = minimize(sum(x))
    m.optimize()

    total = sum(m.value(v) for v in x)
    assert abs(total - 10.0) < 1e-6


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
