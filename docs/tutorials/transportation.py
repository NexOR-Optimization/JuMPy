# Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
r"""
The transportation problem
==========================

This tutorial chooses how many pogo sticks to ship from a set of factories to
a set of stores while minimizing the total shipping cost. Each factory has a
limited supply, each store has a fixed demand, and some shipping routes are not
available.

**This tutorial was originally contributed to JuMP by Louis Luangkesorn.** It
is a Python translation and adaptation of JuMP's
`transportation tutorial <https://github.com/jump-dev/JuMP.jl/blob/b1888e6e9e6f07b03529340216fc07ba95f91fd5/docs/src/tutorials/linear/transp.jl>`_.
The original tutorial was adapted to the current JuMPy API and its compiled
HiGHS backend.

Let :math:`O` be the origins and :math:`D` be the destinations. Origin
:math:`i` has supply :math:`s_i`, destination :math:`j` has demand :math:`d_j`,
and shipping one unit from :math:`i` to :math:`j` costs :math:`c_{i,j}`. The
linear program is

.. math::

   \begin{aligned}
   \min \quad & \sum_{i \in O, j \in D} c_{i,j} x_{i,j} \\
   \text{s.t.} \quad & \sum_{j \in D} x_{i,j} \leq s_i
       && \forall i \in O, \\
   & \sum_{i \in O} x_{i,j} = d_j && \forall j \in D, \\
   & x_{i,j} \geq 0 && \forall i \in O, j \in D.
   \end{aligned}
"""

import jumpy.highs as jp

# Data
# ----
#
# The rows below contain an origin name, one shipping cost per destination,
# and the origin's supply. A ``.`` marks a route that is not available. The
# final row contains the demand at each destination.

table = """
       . FRA  DET LAN WIN  STL  FRE  LAF SUPPLY
    GARY  39   14  11  14   16   82    8   1400
    CLEV  27    .  12   .   26   95   17   2600
    PITT  24   14  17  13   28   99   20   2900
  DEMAND 900 1200 600 400 1700 1100 1000      0
"""

rows = [line.split() for line in table.splitlines() if line.strip()]
destinations = rows[0][1:-1]
origins = [row[0] for row in rows[1:-1]]
cost = rows[1:-1]
supply_values = [float(row[-1]) for row in cost]
demand_values = [float(value) for value in rows[-1][1:-1]]
cost_values = [value for row in cost for value in row[1:-1]]

print(f"Origins: {origins}")
print(f"Destinations: {destinations}")

# Formulation
# -----------
#
# JuMPy currently stores multidimensional variables as a flat vector. The
# helper below maps an origin and destination pair to its position in that
# vector.

n_origins = len(origins)
n_destinations = len(destinations)


def position(origin, destination):
    return n_destinations * origin + destination


model = jp.Model()
x = model.variables(n_origins * n_destinations, lower=0, name="x")

# Missing routes are fixed to zero. The remaining routes form the objective.

for origin in range(n_origins):
    for destination in range(n_destinations):
        index = position(origin, destination)
        if cost_values[index] == ".":
            model.constraint(x[index] == 0)

model.objective = jp.minimize(
    sum(
        float(cost_values[index]) * x[index]
        for index in range(len(x))
        if cost_values[index] != "."
    )
)

# Constraint groups
# -----------------
#
# A symbolic iterator creates every supply constraint from one expression
# template. The supply data is stored as a JuMPy parameter so it can be indexed
# by the same iterator.

supply = model.parameter(supply_values, name="supply")
origin = model.iterator(range(n_origins))
model.constraint_group(
    sum(x[position(origin, destination)] for destination in range(n_destinations))
    <= supply[origin]
)

# A second template creates one demand constraint per destination.

demand = model.parameter(demand_values, name="demand")
destination = model.iterator(range(n_destinations))
model.constraint_group(
    sum(x[position(origin, destination)] for origin in range(n_origins))
    == demand[destination]
)

# Solution
# --------

model.optimize()
shipment = [model.value(variable) for variable in x]

for origin in range(n_origins):
    for destination in range(n_destinations):
        value = shipment[position(origin, destination)]
        if value > 1e-6:
            print(
                f"{origins[origin]} -> {destinations[destination]}: {value:.0f}"
            )

# These checks execute both when the file is run directly and when Sphinx
# builds the gallery, but they are omitted from the rendered tutorial.

# sphinx_gallery_start_ignore
for origin in range(n_origins):
    shipped = sum(
        shipment[position(origin, destination)]
        for destination in range(n_destinations)
    )
    assert shipped <= supply_values[origin] + 1e-6

for destination in range(n_destinations):
    received = sum(
        shipment[position(origin, destination)] for origin in range(n_origins)
    )
    assert abs(received - demand_values[destination]) <= 1e-6

for index, value in enumerate(cost_values):
    if value == ".":
        assert abs(shipment[index]) <= 1e-6

model.close()
# sphinx_gallery_end_ignore
