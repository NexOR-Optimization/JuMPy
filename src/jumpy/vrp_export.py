"""
Exports a JuMPy VRP model to the MathOptVRP JSON format.

Only handles models that contain:
  - A VRPConstraint with a Partition set  (from m.constraint_in_set)
  - A VRPObjective built from op_sum_distances  (from jp.minimize(...))
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.vrp import Partition, VRPObjective, OpSumDistances, _SumOfDistances


def write_vrp_json(model: "Model", filename: str, locations=None) -> None:
    """
    Write a VRP model to a MathOptVRP JSON file.

    Parameters
    ----------
    model     : Model  — must contain a Partition constraint and a VRP objective.
    filename  : str    — output path for the .json file.
    locations : list of [lat, lon] pairs, optional.
                One entry per location (length must equal the dimension of the
                distance matrix used in op_sum_distances).  A location is
                labelled "depot" if it appears as a start or end point in any
                truck's op_sum_distances sequence; all others are "client".
                When omitted the locations entries are written without a
                "coordinates" key.
    """
    with open(filename, "w") as f:
        json.dump(_model_to_vrp_json(model, locations), f, indent=2)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _model_to_vrp_json(model: "Model", locations) -> dict:
    partition_con = _find_partition(model)
    part = partition_con.set
    n = part.n_clients
    k = part.n_trucks

    var_names = _partition_var_names(partition_con.variables, n, k)

    obj = model._objective
    if not isinstance(obj, VRPObjective):
        raise ValueError(
            "Model must have a VRP objective built from op_sum_distances. "
            "Use jp.minimize(sum(op_sum_distances(...) for ...))"
        )

    terms = obj.expr.terms if isinstance(obj.expr, _SumOfDistances) else [obj.expr]

    # Per-truck (start, end) depot indices, extracted from each sequence.
    truck_depots = [_depots_from_term(t) for t in terms]
    depot_ids = {d for start, end in truck_depots for d in (start, end)}

    # Total location count comes from the distance matrix — no assumptions
    # about which IDs are depots.
    n_locations = len(terms[0].matrix)

    return {
        "name": "MathOptVRP Model",
        "problem_type": "VRP", # For the moment assume only classical VRP
        "variables": [{"name": v} for v in var_names],
        "constraints": [_partition_constraint(n, k, var_names)],
        "objective": {"sense": obj.sense},
        "locations": _locations(n_locations, depot_ids, locations),
        "vehicles": [
            {"id": t, "start": start, "end": end}
            for t, (start, end) in enumerate(truck_depots)
        ],
    }


def _find_partition(model: "Model"):
    for con in model._vrp_constraints:
        if isinstance(con.set, Partition):
            return con
    raise ValueError(
        "No Partition constraint found. "
        "Add one with m.constraint_in_set(variables, Partition(n, k))."
    )


def _partition_var_names(var_block, n: int, k: int) -> list[str]:
    """
    Return 2-D MathOptVRP variable names in column-major order.

    Flat index t*n + i (truck t, client slot i, both 0-based) maps to
    the Julia name  base[i+1, t+1]  (1-based row = client, column = truck).
    """
    base = var_block.name or "nodes"
    names = []
    for t in range(k):
        for i in range(n):
            names.append(f"{base}[{i + 1},{t + 1}]")
    return names


def _partition_constraint(n: int, k: int, var_names: list[str]) -> dict:
    return {
        "type": "Partition",
        "flatten": "column-major",
        "num_clients": n,
        "num_trucks": k,
        "variables": var_names,
    }


def _depots_from_term(term: OpSumDistances) -> tuple[int, int]:
    """Return (start_depot, end_depot) for a single op_sum_distances term."""
    seq = term.sequence
    if len(seq) < 2:
        raise ValueError("op_sum_distances sequence must have at least 2 elements")
    for label, val in (("start", seq[0]), ("end", seq[-1])):
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"The {label} of an op_sum_distances sequence must be a depot "
                f"integer, got {type(val).__name__}"
            )
    return int(seq[0]), int(seq[-1])


def _locations(n_locations: int, depot_ids: set, coordinates) -> list[dict]:
    """
    Build the locations list for all location IDs 0..n_locations-1.

    A location is a "depot" if its ID appears in depot_ids, otherwise "client".
    If coordinates is provided it must have exactly n_locations entries.
    """
    if coordinates is not None and len(coordinates) != n_locations:
        raise ValueError(
            f"locations must have {n_locations} entries "
            f"(distance matrix dimension), got {len(coordinates)}"
        )
    entries = []
    for loc_id in range(n_locations):
        role = "depot" if loc_id in depot_ids else "client"
        entry: dict = {"id": loc_id, "role": role}
        if coordinates is not None:
            entry["coordinates"] = list(coordinates[loc_id])
        entries.append(entry)
    return entries
