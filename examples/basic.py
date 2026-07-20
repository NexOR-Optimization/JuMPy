"""
Basic JuMPy usage example.

Every operation is one MOI call into the backend; constraint groups are
expanded by GenOpt in compiled Julia, not in Python.
"""

import sys
sys.path.insert(0, "src")

import jumpy as jp

# ── Create a model ────────────────────────────────────────────────────────────

m = jp.Model()
x = m.variables(100, lower=0, name="x")

# ── Constraint group: consecutive variable pairs ──────────────────────────────
# Instead of 99 individual constraints built in Python (slow!),
# we define ONE template. GenOpt expands it in compiled Julia.

i = m.iterator(range(99))
m.constraint_group(x[i] + x[i + 1] <= 10)

# ── Multi-dimensional constraint group ────────────────────────────────────────

p = m.iterator(range(10))
q = m.iterator(range(10))
m.constraint_group(x[10 * p + q] >= 0)

# ── Constraint group with data parameters ─────────────────────────────────────

costs = m.parameter([float(k) * 0.5 + 1.0 for k in range(100)], name="costs")
k = m.iterator(range(100))
m.constraint_group(costs[k] * x[k] <= 50)

# ── Objective and solve ───────────────────────────────────────────────────────

m.objective = jp.minimize(x[0] + x[1])
m.optimize()

print("x[0] =", m.value(x[0]))
print("x[1] =", m.value(x[1]))
