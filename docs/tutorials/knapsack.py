# Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
r"""
The knapsack problem
====================

This tutorial demonstrates how to formulate and solve a binary integer linear
program with JuMPy. Given a collection of items and a capacity, we choose the
items with the greatest combined profit whose combined weight fits within the
capacity.

This is a Python translation and adaptation of JuMP's
`knapsack tutorial <https://github.com/jump-dev/JuMP.jl/blob/b1888e6e9e6f07b03529340216fc07ba95f91fd5/docs/src/tutorials/linear/knapsack.jl>`_.
The original tutorial was adapted to the current JuMPy API and its compiled
HiGHS backend.

The model is

.. math::

   \begin{aligned}
   \max \quad & \sum_{i=1}^n c_i x_i \\
   \text{s.t.} \quad & \sum_{i=1}^n w_i x_i \leq C, \\
   & x_i \in \{0, 1\}, \quad i = 1, \ldots, n,
   \end{aligned}

where :math:`C` is the capacity and item :math:`i` has profit :math:`c_i` and
weight :math:`w_i`.
"""

import jumpy as jp

# Data
# ----
#
# Our example has five items and a capacity of 10 units.

profit = [5.0, 3.0, 2.0, 7.0, 4.0]
weight = [2.0, 8.0, 4.0, 2.0, 5.0]
capacity = 10.0
n = len(weight)

# Formulation
# -----------
#
# A binary variable records whether each item is selected.

model = jp.Model()
x = model.variables(n, binary=True, name="x")

# JuMPy expressions use ordinary Python arithmetic. Here, ``sum`` constructs
# the capacity constraint and objective one term at a time.

model.constraint(sum(weight[i] * x[i] for i in range(n)) <= capacity)
model.objective = jp.maximize(sum(profit[i] * x[i] for i in range(n)))

# Solution
# --------

model.optimize()
items_chosen = [i for i in range(n) if model.value(x[i]) > 0.5]
total_profit = sum(profit[i] for i in items_chosen)

print(f"Chosen item indices: {items_chosen}")
print(f"Total profit: {total_profit}")

# The assertions below are executed when the script or documentation runs, but
# they are removed from the rendered tutorial.

# sphinx_gallery_start_ignore
assert items_chosen == [0, 3, 4]
assert total_profit == 16.0
model.close()
# sphinx_gallery_end_ignore
