# Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
r"""
The facility location problem
=============================

This tutorial decides which facilities to open and assigns each client to an
open facility. Opening a facility incurs a fixed cost, while serving a client
incurs a cost proportional to the distance between the client and facility.

**This tutorial was originally contributed to JuMP by Mathieu Tanneau and
Alexis Montoison.** It is a Python translation and adaptation of JuMP's
`facility-location tutorial <https://github.com/jump-dev/JuMP.jl/blob/b1888e6e9e6f07b03529340216fc07ba95f91fd5/docs/src/tutorials/linear/facility_location.jl>`_.
This version covers the uncapacitated formulation, uses fixed data instead of
random data, and omits plotting so that it remains fast and dependency-free.

Let :math:`M` be the clients and :math:`N` the possible facility locations.
Binary variable :math:`y_j` records whether facility :math:`j` is open, and
binary variable :math:`x_{i,j}` records whether client :math:`i` is assigned to
facility :math:`j`. Given assignment costs :math:`c_{i,j}` and opening costs
:math:`f_j`, the model is

.. math::

   \begin{aligned}
   \min \quad & \sum_{i \in M, j \in N} c_{i,j} x_{i,j}
                 + \sum_{j \in N} f_j y_j \\
   \text{s.t.} \quad & \sum_{j \in N} x_{i,j} = 1
                         && \forall i \in M, \\
   & x_{i,j} \leq y_j && \forall i \in M, j \in N, \\
   & x_{i,j}, y_j \in \{0, 1\}.
   \end{aligned}
"""

from math import hypot

import jumpy as jp

# Data
# ----
#
# Each tuple is a point in the plane. The assignment cost is the Euclidean
# distance from a client to a facility. The opening costs make it worthwhile to
# share facilities instead of opening the closest one for every client.

client_locations = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (8.0, 8.0),
    (8.0, 9.0),
    (9.0, 8.0),
]
facility_locations = [(0.0, 0.0), (4.5, 4.5), (9.0, 9.0)]
opening_cost = [2.0, 2.0, 2.0]

n_clients = len(client_locations)
n_facilities = len(facility_locations)

assignment_cost = [
    hypot(client_x - facility_x, client_y - facility_y)
    for client_x, client_y in client_locations
    for facility_x, facility_y in facility_locations
]


def position(client, facility):
    """Return the flat index corresponding to a client-facility pair."""
    return n_facilities * client + facility


# Formulation
# -----------

model = jp.Model()
is_open = model.variables(n_facilities, binary=True, name="is_open")
is_assigned = model.variables(
    n_clients * n_facilities,
    binary=True,
    name="is_assigned",
)

# Each client is assigned to exactly one facility. A single symbolic iterator
# expands this expression into one constraint per client.

client = model.iterator(range(n_clients))
model.constraint_group(
    sum(
        is_assigned[position(client, facility)]
        for facility in range(n_facilities)
    )
    == 1
)

# An assignment is allowed only when its facility is open. This template uses
# two iterators, so GenOpt expands it over every client-facility pair.

facility = model.iterator(range(n_facilities))
model.constraint_group(
    is_assigned[position(client, facility)] <= is_open[facility]
)

model.objective = jp.minimize(
    sum(opening_cost[j] * is_open[j] for j in range(n_facilities))
    + sum(
        assignment_cost[index] * is_assigned[index]
        for index in range(len(is_assigned))
    )
)

# Solution
# --------

model.optimize()

open_facilities = [
    facility
    for facility in range(n_facilities)
    if model.value(is_open[facility]) > 0.5
]
assignments = [
    next(
        facility
        for facility in range(n_facilities)
        if model.value(is_assigned[position(client, facility)]) > 0.5
    )
    for client in range(n_clients)
]

print(f"Open facilities: {open_facilities}")
for client, facility in enumerate(assignments):
    print(f"Client {client} is served by facility {facility}")

# These checks execute when the script or documentation runs, but are omitted
# from the rendered tutorial.

# sphinx_gallery_start_ignore
assert len(assignments) == n_clients
assert all(facility in open_facilities for facility in assignments)
assert assignments[:3] == [0, 0, 0]
assert assignments[3:] == [2, 2, 2]
assert open_facilities == [0, 2]
model.close()
# sphinx_gallery_end_ignore
