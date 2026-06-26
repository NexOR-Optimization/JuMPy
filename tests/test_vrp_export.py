"""
Structural tests for JuMPy's VRP JSON exporter (vrp_export.py).

Tests inspect the JSON produced by write_vrp_json without solving anything.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jumpy as jp
from jumpy.vrp import Partition, op_sum_distances

_TMP = tempfile.mkdtemp(prefix="vrp_export_test_")

# 7 locations: depot at 0, clients at 1..6
COORDS_SINGLE = [
    [4.3517, 50.8466],  # depot (id=0)
    [4.4025, 50.8550],
    [4.3300, 50.8400],
    [4.3700, 50.8300],
    [4.3900, 50.8700],
    [4.3450, 50.8600],
    [4.4100, 50.8500],
]

# 8 locations: depots at 0 and 1, clients at 2..7
COORDS_MULTI = [
    [4.3517, 50.8466],  # depot A (id=0)
    [4.3600, 50.8500],  # depot B (id=1)
    [4.4025, 50.8550],
    [4.3300, 50.8400],
    [4.3700, 50.8300],
    [4.3900, 50.8700],
    [4.3450, 50.8600],
    [4.4100, 50.8500],
]


def _build_single_depot_model(n=6, k=2, depot=0):
    """Single depot: all trucks start and end at depot."""
    N = n + 1  # matrix dimension: 1 depot + n clients
    M = [[abs(i - j) * 100 for j in range(N)] for i in range(N)]
    m = jp.Model(backend="vroom")
    nodes = m.constraint_in_set(m.variables(n * k, name="nodes"), Partition(n, k))
    m.objective = jp.minimize(
        sum(
            op_sum_distances(M, [depot] + [nodes[t * n + i] for i in range(n)] + [depot])
            for t in range(k)
        )
    )
    return m


def _build_multi_depot_model(n=6, k=2, depot_a=0, depot_b=1):
    """Multi-depot: truck 0 uses depot_a, truck 1 uses depot_b."""
    N = n + 2  # matrix dimension: 2 depots + n clients
    M = [[abs(i - j) * 100 for j in range(N)] for i in range(N)]
    m = jp.Model(backend="vroom")
    nodes = m.constraint_in_set(m.variables(n * k, name="nodes"), Partition(n, k))
    depots = [depot_a, depot_b]
    m.objective = jp.minimize(
        sum(
            op_sum_distances(
                M,
                [depots[t]] + [nodes[t * n + i] for i in range(n)] + [depots[t]],
            )
            for t in range(k)
        )
    )
    return m


def _export(m, name, locations=None):
    path = os.path.join(_TMP, f"{name}.json")
    m.write_vrp_json(path, locations=locations)
    with open(path) as f:
        return json.load(f)


# ── single-depot tests ────────────────────────────────────────────────────────

def test_top_level_fields():
    doc = _export(_build_single_depot_model(), "top_level")
    assert doc["format_version"] == 2
    assert doc["problem_type"] == "VRP"
    assert doc["name"] == "MathOptVRP Model"
    assert "variables" in doc
    assert "constraints" in doc
    assert "locations" in doc
    assert "vehicles" in doc
    assert doc["objective"] == {"sense": "min"}
    print("PASS test_top_level_fields")


def test_variable_names_2d():
    doc = _export(_build_single_depot_model(n=6, k=2), "var_names")
    names = [v["name"] for v in doc["variables"]]
    expected = (
        [f"nodes[{i},1]" for i in range(1, 7)]
        + [f"nodes[{i},2]" for i in range(1, 7)]
    )
    assert names == expected, f"got {names}"
    print("PASS test_variable_names_2d")


def test_partition_constraint():
    doc = _export(_build_single_depot_model(n=6, k=2), "partition")
    cons = doc["constraints"]
    assert len(cons) == 1
    c = cons[0]
    assert c["type"] == "Partition"
    assert c["flatten"] == "column-major"
    assert c["num_clients"] == 6
    assert c["num_trucks"] == 2
    assert c["variables"] == [v["name"] for v in doc["variables"]]
    print("PASS test_partition_constraint")


def test_single_depot_vehicles():
    doc = _export(_build_single_depot_model(n=6, k=2, depot=0), "vehicles_single")
    vehicles = doc["vehicles"]
    assert len(vehicles) == 2
    for t, v in enumerate(vehicles):
        assert v["id"] == t
        assert v["start"] == 0
        assert v["end"] == 0
    print("PASS test_single_depot_vehicles")


def test_locations_without_coordinates():
    doc = _export(_build_single_depot_model(n=6, k=2), "no_coords")
    locs = doc["locations"]
    assert len(locs) == 7  # 1 depot + 6 clients (matrix is 7x7)
    assert locs[0] == {"id": 0, "role": "depot"}
    for i in range(1, 7):
        assert locs[i] == {"id": i, "role": "client"}
    print("PASS test_locations_without_coordinates")


def test_locations_with_coordinates():
    doc = _export(_build_single_depot_model(n=6, k=2), "with_coords", locations=COORDS_SINGLE)
    locs = doc["locations"]
    assert len(locs) == 7
    for i, loc in enumerate(locs):
        assert loc["coordinates"] == COORDS_SINGLE[i]
    print("PASS test_locations_with_coordinates")


def test_wrong_location_count_raises():
    m = _build_single_depot_model(n=6, k=2)
    try:
        m.write_vrp_json(os.path.join(_TMP, "bad.json"), locations=[[0, 0]] * 3)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS test_wrong_location_count_raises")


def test_no_vrp_objective_raises():
    m = jp.Model(backend="vroom")
    m.constraint_in_set(m.variables(4, name="nodes"), Partition(2, 2))
    m.objective = jp.minimize(jp.Constant(0))
    try:
        m.write_vrp_json(os.path.join(_TMP, "bad_obj.json"))
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS test_no_vrp_objective_raises")


# ── multi-depot tests ─────────────────────────────────────────────────────────

def test_multi_depot_vehicles():
    doc = _export(_build_multi_depot_model(n=6, k=2, depot_a=0, depot_b=1), "vehicles_multi")
    vehicles = doc["vehicles"]
    assert len(vehicles) == 2
    assert vehicles[0] == {"id": 0, "start": 0, "end": 0}
    assert vehicles[1] == {"id": 1, "start": 1, "end": 1}
    print("PASS test_multi_depot_vehicles")


def test_multi_depot_location_roles():
    doc = _export(_build_multi_depot_model(n=6, k=2, depot_a=0, depot_b=1), "roles_multi")
    locs = doc["locations"]
    assert len(locs) == 8  # 2 depots + 6 clients (matrix is 8x8)
    assert locs[0]["role"] == "depot"
    assert locs[1]["role"] == "depot"
    for i in range(2, 8):
        assert locs[i]["role"] == "client"
    print("PASS test_multi_depot_location_roles")


def test_multi_depot_with_coordinates():
    doc = _export(
        _build_multi_depot_model(n=6, k=2, depot_a=0, depot_b=1),
        "coords_multi",
        locations=COORDS_MULTI,
    )
    locs = doc["locations"]
    assert len(locs) == 8
    for i, loc in enumerate(locs):
        assert loc["coordinates"] == COORDS_MULTI[i]
    print("PASS test_multi_depot_with_coordinates")


def test_asymmetric_depot_start_end():
    """Truck starts at depot 0, ends at depot 1."""
    n, k = 4, 1
    N = n + 2  # 2 depots + 4 clients
    M = [[abs(i - j) * 10 for j in range(N)] for i in range(N)]
    m = jp.Model(backend="vroom")
    nodes = m.constraint_in_set(m.variables(n * k, name="nodes"), Partition(n, k))
    m.objective = jp.minimize(
        op_sum_distances(M, [0] + [nodes[i] for i in range(n)] + [1])
    )
    doc = _export(m, "asymmetric")
    assert doc["vehicles"] == [{"id": 0, "start": 0, "end": 1}]
    assert doc["locations"][0]["role"] == "depot"
    assert doc["locations"][1]["role"] == "depot"
    for i in range(2, N):
        assert doc["locations"][i]["role"] == "client"
    print("PASS test_asymmetric_depot_start_end")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_top_level_fields,
        test_variable_names_2d,
        test_partition_constraint,
        test_single_depot_vehicles,
        test_locations_without_coordinates,
        test_locations_with_coordinates,
        test_wrong_location_count_raises,
        test_no_vrp_objective_raises,
        test_multi_depot_vehicles,
        test_multi_depot_location_roles,
        test_multi_depot_with_coordinates,
        test_asymmetric_depot_start_end,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
