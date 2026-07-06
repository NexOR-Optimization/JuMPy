"""
Structural tests for JuMPy's MOF exporter (mof_export.py).

These tests inspect the JSON that `write_mof` produces, they do not solve
the optimization problem. They verify that every kind of
model element (variables, bounds, integrality, affine/nonlinear objectives and constraints, 
constraint groups, parameters) is serialized into a correct and
schema-consistent MathOptFormat document.
"""

import json
import math
import os
import tempfile

import jumpy as jp
from jumpy.expressions import sin, cos, exp, log, sqrt
from jumpy.vrp import op_sum_distances

_TMP = tempfile.mkdtemp(prefix="mof_test_")

# -- helpers ----------------------------------------------------------------

def export(m, name):
    """Write the model to MOF and return the parsed JSON document."""
    path = os.path.join(_TMP, f"{name}.mof.json")
    m.write_mof(path)
    with open(path) as f:
        return json.load(f)


def real_constraints(doc):
    """Constraints that are not variable bounds/integrality (i.e. the model's
    actual functional constraints)."""
    return [c for c in doc["constraints"]
            if c["function"].get("type") != "Variable"]


def bound_constraints(doc, set_type=None):
    """Variable-bound / integrality constraints, optionally filtered by set."""
    out = [c for c in doc["constraints"]
           if c["function"].get("type") == "Variable"]
    if set_type is not None:
        out = [c for c in out if c["set"]["type"] == set_type]
    return out


def obj_terms(doc):
    """{var_name: coefficient} for an affine objective."""
    return {t["variable"]: t["coefficient"]
            for t in doc["objective"]["function"]["terms"]}


def con_terms(c):
    """{var_name: coefficient} for an affine constraint entry."""
    return {t["variable"]: t["coefficient"] for t in c["function"]["terms"]}


# Document structure

def test_document_has_required_top_level_keys():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0])
    doc = export(m, "doc_keys")
    for key in ("name", "version", "variables", "objective", "constraints"):
        assert key in doc, f"missing top-level key '{key}'"


def test_version_is_mof_1_7():
    m = jp.Model(backend="juliacall")
    m.variables(1, name="x")
    doc = export(m, "version")
    assert doc["version"] == {"major": 1, "minor": 7}, doc["version"]


def test_variable_count_matches():
    m = jp.Model(backend="juliacall")
    m.variables(5, name="x")
    m.variables(3, name="y")
    doc = export(m, "varcount")
    assert len(doc["variables"]) == 8


# Variables and bounds

def test_lower_bound_emits_greaterthan():
    m = jp.Model(backend="juliacall")
    m.variables(2, lower=0, name="x")
    doc = export(m, "lower")
    gts = bound_constraints(doc, "GreaterThan")
    assert len(gts) == 2
    assert all(c["set"]["lower"] == 0.0 for c in gts)


def test_upper_bound_emits_lessthan():
    m = jp.Model(backend="juliacall")
    m.variables(2, upper=10, name="x")
    doc = export(m, "upper")
    lts = bound_constraints(doc, "LessThan")
    assert len(lts) == 2
    assert all(c["set"]["upper"] == 10.0 for c in lts)


def test_both_bounds_emit_two_constraints_each():
    m = jp.Model(backend="juliacall")
    m.variables(2, lower=1, upper=5, name="x")
    doc = export(m, "both")
    assert len(bound_constraints(doc, "GreaterThan")) == 2
    assert len(bound_constraints(doc, "LessThan")) == 2


def test_no_bounds_emits_no_bound_constraints():
    m = jp.Model(backend="juliacall")
    m.variables(3, name="x")
    m.objective = jp.minimize(jp.Constant(0))
    doc = export(m, "nobounds")
    assert len(bound_constraints(doc)) == 0


def test_unnamed_variables_get_fallback_name():
    m = jp.Model(backend="juliacall")
    m.variables(2)  # no name
    doc = export(m, "unnamed")
    names = {v["name"] for v in doc["variables"]}
    # fallback is v{index}
    assert names == {"v0", "v1"}, names


def test_named_variables_use_given_name():
    m = jp.Model(backend="juliacall")
    m.variables(3, name="route")
    doc = export(m, "named")
    names = [v["name"] for v in doc["variables"]]
    assert names == ["route[0]", "route[1]", "route[2]"]


# Integrality

def test_binary_emits_zeroone():
    m = jp.Model(backend="juliacall")
    m.variables(4, binary=True, name="x")
    doc = export(m, "binary")
    assert len(bound_constraints(doc, "ZeroOne")) == 4


def test_integer_emits_integer():
    m = jp.Model(backend="juliacall")
    m.variables(3, integer=True, name="x")
    doc = export(m, "integer")
    assert len(bound_constraints(doc, "Integer")) == 3


def test_binary_with_bounds_coexist():
    m = jp.Model(backend="juliacall")
    m.variables(2, lower=0, upper=1, binary=True, name="x")
    doc = export(m, "binbound")
    assert len(bound_constraints(doc, "ZeroOne")) == 2
    assert len(bound_constraints(doc, "GreaterThan")) == 2
    assert len(bound_constraints(doc, "LessThan")) == 2


def test_zeroone_references_correct_variables():
    m = jp.Model(backend="juliacall")
    m.variables(2, binary=True, name="b")
    doc = export(m, "binnames")
    names = {c["function"]["name"] for c in bound_constraints(doc, "ZeroOne")}
    assert names == {"b[0]", "b[1]"}


def test_continuous_block_not_affected_by_neighbour_binary_block():
    m = jp.Model(backend="juliacall")
    m.variables(2, binary=True, name="b")
    m.variables(2, lower=0, name="c")  # continuous
    doc = export(m, "mixedblocks")
    assert len(bound_constraints(doc, "ZeroOne")) == 2  # only b
    assert len(bound_constraints(doc, "GreaterThan")) == 2  # only c


# Affine objectives

def test_no_objective_is_feasibility():
    m = jp.Model(backend="juliacall")
    m.variables(2, name="x")
    doc = export(m, "feasibility")
    assert doc["objective"]["sense"] == "feasibility"
    assert "function" not in doc["objective"]


def test_minimize_sense():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0] + x[1])
    doc = export(m, "min")
    assert doc["objective"]["sense"] == "min"


def test_maximize_sense():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.maximize(x[0] + x[1])
    doc = export(m, "max")
    assert doc["objective"]["sense"] == "max"


def test_affine_objective_coefficients():
    m = jp.Model(backend="juliacall")
    x = m.variables(3, name="x")
    m.objective = jp.minimize(2*x[0] + 3*x[1] - x[2])
    doc = export(m, "affcoef")
    assert doc["objective"]["function"]["type"] == "ScalarAffineFunction"
    t = obj_terms(doc)
    assert t["x[0]"] == 2.0
    assert t["x[1]"] == 3.0
    assert t["x[2]"] == -1.0


def test_objective_constant_term_preserved():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0] + 7)
    doc = export(m, "objconst")
    assert doc["objective"]["function"]["constant"] == 7.0


def test_objective_coefficients_aggregate():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0] + x[0] + x[0])
    doc = export(m, "objaggregate")
    assert obj_terms(doc)["x[0]"] == 3.0


def test_objective_zero_coefficient_filtered():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0] - x[0] + x[1])
    doc = export(m, "objzero")
    t = obj_terms(doc)
    assert "x[0]" not in t  # zero coefficient dropped
    assert t["x[1]"] == 1.0


def test_objective_sum_over_stays_affine():
    m = jp.Model(backend="juliacall")
    x = m.variables(4, name="x")
    costs = jp.Parameter([1.0, 2.0, 3.0, 4.0])
    i = jp.Iterator(range(4))
    m.objective = jp.minimize(jp.sum_over(i, costs[i] * x[i]))
    doc = export(m, "objsumover")
    assert doc["objective"]["function"]["type"] == "ScalarAffineFunction"
    t = obj_terms(doc)
    assert [t[f"x[{k}]"] for k in range(4)] == [1.0, 2.0, 3.0, 4.0]


# Nonlinear objectives

def test_exp_objective_is_nonlinear():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(exp(x[0]))
    doc = export(m, "objexp")
    assert doc["objective"]["function"]["type"] == "ScalarNonlinearFunction"


def test_quadratic_objective_is_nonlinear():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0] * x[1])  # bilinear → nonlinear
    doc = export(m, "objbilinear")
    assert doc["objective"]["function"]["type"] == "ScalarNonlinearFunction"


def test_power_objective_is_nonlinear():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0]**2)
    doc = export(m, "objpow")
    assert doc["objective"]["function"]["type"] == "ScalarNonlinearFunction"


def test_each_named_function_maps_to_operator():
    """sin/cos/exp/log/sqrt all appear as operator nodes."""
    fns = {"sin": sin, "cos": cos, "exp": exp, "log": log, "sqrt": sqrt}
    for opname, fn in fns.items():
        m = jp.Model(backend="juliacall")
        x = m.variables(1, name="x")
        m.objective = jp.minimize(fn(x[0]))
        doc = export(m, f"obj_{opname}")
        root = doc["objective"]["function"]["root"]
        ops = _collect_op_types(doc["objective"]["function"])
        assert opname in ops, f"operator '{opname}' not found in {ops}"


# Affine constraints (individual)

def test_affine_leq_constant_folded():
    # 2x + 5 <= 11  ->  2x <= 6  (constant folded into bound)
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(2*x[0] + 5 <= 11)
    doc = export(m, "leq")
    c = real_constraints(doc)[0]
    assert c["function"]["type"] == "ScalarAffineFunction"
    assert c["function"]["constant"] == 0.0
    assert c["set"]["type"] == "LessThan"
    assert c["set"]["upper"] == 6.0


def test_affine_geq():
    # x + y >= 3
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(x[0] + x[1] >= 3)
    doc = export(m, "geq")
    c = real_constraints(doc)[0]
    assert c["set"]["type"] == "GreaterThan"
    assert c["set"]["lower"] == 3.0


def test_affine_equality():
    # 2x + y == 6
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(2*x[0] + x[1] == 6)
    doc = export(m, "eq")
    c = real_constraints(doc)[0]
    assert c["set"]["type"] == "EqualTo"
    assert c["set"]["value"] == 6.0
    assert con_terms(c)["x[0]"] == 2.0


def test_constraint_with_variables_on_both_sides():
    # x <= y  ->  x - y <= 0
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(x[0] <= x[1])
    doc = export(m, "bothsides")
    c = real_constraints(doc)[0]
    t = con_terms(c)
    assert t["x[0]"] == 1.0
    assert t["x[1]"] == -1.0
    assert c["set"]["upper"] == 0.0


# Nonlinear constraints (individual)

def test_nonlinear_constraint_against_zero():
    # sin(x) + 1 <= 2  ->  nonlinear, set LessThan(0)
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(sin(x[0]) + 1 <= 2)
    doc = export(m, "nlcon")
    c = real_constraints(doc)[0]
    assert c["function"]["type"] == "ScalarNonlinearFunction"
    assert c["set"]["type"] == "LessThan"
    assert c["set"]["upper"] == 0.0


def test_bilinear_constraint_is_nonlinear():
    # x*y <= 4
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(x[0] * x[1] <= 4)
    doc = export(m, "bilincon")
    c = real_constraints(doc)[0]
    assert c["function"]["type"] == "ScalarNonlinearFunction"


def test_nonlinear_equality_constraint():
    # exp(x) == e^2
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0])
    m.constraint(exp(x[0]) == math.e**2)
    doc = export(m, "nleq")
    c = real_constraints(doc)[0]
    assert c["function"]["type"] == "ScalarNonlinearFunction"
    assert c["set"]["type"] == "EqualTo"
    assert c["set"]["value"] == 0.0


# Constraint groups

def test_single_iterator_group_expands():
    # x[i] <= 5 for i in 0..3  -> 4 constraints
    m = jp.Model(backend="juliacall")
    x = m.variables(4, name="x")
    m.objective = jp.minimize(x[0])
    i = jp.Iterator(range(4))
    m.constraint_group([i], x[i] <= 5)
    doc = export(m, "group1")
    assert len(real_constraints(doc)) == 4
    for c in real_constraints(doc):
        assert c["set"]["type"] == "LessThan"
        assert c["set"]["upper"] == 5.0


def test_two_iterator_group_expands_product():
    # x[3i+j] >= 0 for i in 0..1, j in 0..2  -> 6 constraints
    m = jp.Model(backend="juliacall")
    x = m.variables(6, name="x")
    m.objective = jp.minimize(x[0])
    i = jp.Iterator(range(2))
    j = jp.Iterator(range(3))
    m.constraint_group([i, j], x[3*i + j] >= 0)
    doc = export(m, "group2")
    assert len(real_constraints(doc)) == 6


def test_group_with_parameter():
    # lb[i] <= x[i]  for i in 0..3
    m = jp.Model(backend="juliacall")
    x = m.variables(4, name="x")
    m.objective = jp.minimize(x[0])
    lb = jp.Parameter([1.0, 2.0, 3.0, 4.0])
    i = jp.Iterator(range(4))
    m.constraint_group([i], x[i] >= lb[i])
    doc = export(m, "groupparam")
    cons = real_constraints(doc)
    bounds = sorted(c["set"]["lower"] for c in cons)
    assert bounds == [1.0, 2.0, 3.0, 4.0]


def test_nonlinear_group_expands_to_nonlinear():
    # exp(x[i]) <= e^3 for i in 0..2
    m = jp.Model(backend="juliacall")
    x = m.variables(3, name="x")
    m.objective = jp.minimize(x[0])
    i = jp.Iterator(range(3))
    m.constraint_group([i], exp(x[i]) <= math.e**3)
    doc = export(m, "nlgroup")
    cons = real_constraints(doc)
    assert len(cons) == 3
    assert all(c["function"]["type"] == "ScalarNonlinearFunction" for c in cons)


def test_group_constraints_have_names():
    m = jp.Model(backend="juliacall")
    x = m.variables(3, name="x")
    m.objective = jp.minimize(x[0])
    i = jp.Iterator(range(3))
    m.constraint_group([i], x[i] <= 5)
    doc = export(m, "groupnames")
    named = [c for c in real_constraints(doc) if "name" in c]
    assert len(named) == 3  # each expanded constraint is named cg0_*


# Nonlinear node structure validity

def _collect_op_types(func):
    """Gather all operator 'type' strings from a ScalarNonlinearFunction."""
    types = set()

    def visit(node):
        if isinstance(node, dict):
            t = node.get("type")
            if t == "node":
                return  # reference
            if t is not None:
                types.add(t)
            for a in node.get("args", []):
                visit(a)
    visit(func["root"])
    for node in func.get("node_list", []):
        visit(node)
    return types


def test_nonlinear_root_is_operator_object():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, name="x")
    m.objective = jp.minimize(sin(x[0]) + x[1]**2)
    doc = export(m, "nlroot")
    func = doc["objective"]["function"]
    root = func["root"]
    if isinstance(root, dict) and root.get("type") == "node":
        # MOI's MOF writer emits the root as a 1-based node_list reference
        root = func["node_list"][root["index"] - 1]
    assert isinstance(root, dict)
    assert "type" in root and "args" in root


def test_nonlinear_node_references_are_1_based_and_valid():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(exp(sin(x[0])))
    doc = export(m, "nlnested")
    func = doc["objective"]["function"]
    n = len(func["node_list"])

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "node":
                idx = node["index"]
                assert 1 <= idx <= n, f"node index {idx} out of range 1..{n}"
            for a in node.get("args", []):
                check(a)
    check(func["root"])
    for node in func["node_list"]:
        check(node)


def test_nonlinear_variable_leaves_are_strings():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(sin(x[0]))
    doc = export(m, "nlleaf")
    func = doc["objective"]["function"]
    # the sin node's arg should be the bare string "x[0]"
    ops = func["node_list"] + [func["root"]]
    found_var = any(
        isinstance(a, str)
        for node in ops if isinstance(node, dict)
        for a in node.get("args", [])
    )
    assert found_var, "expected a variable name string as a leaf"


def test_nonlinear_constant_leaves_are_numbers():
    m = jp.Model(backend="juliacall")
    x = m.variables(1, name="x")
    m.objective = jp.minimize(x[0]**2)
    doc = export(m, "nlconstleaf")
    ops = _collect_op_types(doc["objective"]["function"])
    assert "^" in ops


# Mixed / integration shapes

def test_mixed_affine_and_nonlinear_constraints_coexist():
    m = jp.Model(backend="juliacall")
    x = m.variables(2, lower=0, name="x")
    m.objective = jp.minimize(exp(x[0]) + x[1])    # nonlinear obj
    m.constraint(x[0] + x[1] >= 3)                 # affine con
    m.constraint(sin(x[1]) <= 0.9)                 # nonlinear con
    doc = export(m, "mixed")
    cons = real_constraints(doc)
    types = sorted(c["function"]["type"] for c in cons)
    assert types == ["ScalarAffineFunction", "ScalarNonlinearFunction"]
    assert doc["objective"]["function"]["type"] == "ScalarNonlinearFunction"


# VRP models — the same generic write_mof handles Partition set constraints
# and op_sum_distances objectives through the VectorSet / SolverFunction
# extension protocols.

def _vrp_model(n=3, k=2, depot=1):
    # depot is a 1-based location id: location 1 is the depot, 2..N the clients
    N = n + 1  # 1 depot + n clients
    M = [[abs(i - j) * 100 for j in range(N)] for i in range(N)]
    m = jp.Model(backend="vroom")
    nodes = m.variables(n * k, name="nodes")
    m.constraint_in_set(nodes, jp.Partition(n, k))
    m.objective = jp.minimize(
        sum(
            op_sum_distances(M, [depot] + [nodes[t * n + i] for i in range(n)] + [depot])
            for t in range(k)
        )
    )
    return m


def test_vrp_partition_constraint_exported():
    doc = export(_vrp_model(n=3, k=2), "vrp_partition")
    parts = [c for c in doc["constraints"] if c["set"]["type"] == "Partition"]
    assert len(parts) == 1
    c = parts[0]
    assert c["function"]["type"] == "VectorOfVariables"
    assert len(c["function"]["variables"]) == 6
    assert c["set"]["num_clients"] == 3
    assert c["set"]["num_trucks"] == 2


def test_vrp_variable_names_are_2d_one_based():
    # Partition variables use Julia's `nodes[i,t]` naming, column-major
    doc = export(_vrp_model(n=3, k=2), "vrp_names")
    names = [v["name"] for v in doc["variables"]]
    expected = [f"nodes[{i},{t}]" for t in (1, 2) for i in (1, 2, 3)]
    assert names == expected, names
    parts = [c for c in doc["constraints"] if c["set"]["type"] == "Partition"]
    assert parts[0]["function"]["variables"] == expected


def test_vrp_sequence_depots_pass_through():
    # depot ids are 1-based location indices and are exported unchanged
    doc = export(_vrp_model(n=3, k=2, depot=1), "vrp_depots")
    func = doc["objective"]["function"]
    seqs = [nd for nd in func["node_list"]
            if isinstance(nd, dict) and nd.get("type") == "sequence"]
    assert len(seqs) == 2
    for seq in seqs:
        assert seq["args"][0] == 1.0, seq["args"]
        assert seq["args"][-1] == 1.0, seq["args"]


def test_vrp_has_scalar_nonlinear_flag():
    doc = export(_vrp_model(n=3, k=2), "vrp_snl_flag")
    assert doc.get("has_scalar_nonlinear") is True


def test_vrp_objective_is_sum_of_sum_distances():
    doc = export(_vrp_model(n=3, k=2), "vrp_obj")
    func = doc["objective"]["function"]
    assert func["type"] == "ScalarNonlinearFunction"
    ops = _collect_op_types(func)
    assert "sum_distances" in ops, ops
    assert "sequence" in ops, ops
    assert "matrix" in ops, ops


def test_vrp_three_trucks_sum():
    # regression: sum() over 3+ op_sum_distances terms used to TypeError
    doc = export(_vrp_model(n=2, k=3), "vrp_three")
    func = doc["objective"]["function"]
    seqs = [nd for nd in func["node_list"]
            if isinstance(nd, dict) and nd.get("type") == "sequence"]
    assert len(seqs) == 3, f"expected 3 sequences, got {len(seqs)}"


def test_vrp_mixes_with_standard_variables():
    # a VRP block and an unrelated standard variable coexist in one model
    m = _vrp_model(n=2, k=2)
    m.variables(1, lower=0, name="y")
    doc = export(m, "vrp_mixed")
    assert len(doc["variables"]) == 5  # 4 route slots + y
    assert any(c["set"]["type"] == "Partition" for c in doc["constraints"])


def test_constraint_in_set_dimension_mismatch_raises():
    m = jp.Model(backend="vroom")
    nodes = m.variables(5, name="nodes")
    try:
        m.constraint_in_set(nodes, jp.Partition(2, 2))  # dimension 4 != 5
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_full_milp_shape():
    # binary objective + affine equality + affine inequality
    m = jp.Model(backend="juliacall")
    x = m.variables(3, binary=True, name="x")
    m.objective = jp.minimize(2*x[0] + 3*x[1] + 4*x[2])
    m.constraint(x[0] + x[1] + x[2] == 2)
    m.constraint(x[0] + x[1] <= 1)
    doc = export(m, "milp")
    assert len(bound_constraints(doc, "ZeroOne")) == 3
    cons = real_constraints(doc)
    assert len(cons) == 2
    assert doc["objective"]["function"]["type"] == "ScalarAffineFunction"


if __name__ == "__main__":
    tests = [(n, globals()[n]) for n in sorted(globals()) if n.startswith("test_")]
    passed = failed = 0
    failures = []
    for name, test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            failures.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"  {passed} passed, {failed} failed")
    if failures:
        print("\n  Failures:")
        for name, e in failures:
            print(f"    {name}: {e}")
    raise SystemExit(1 if failed else 0)