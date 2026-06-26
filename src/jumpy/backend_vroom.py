"""
backend_vroom.py
================
JuMPy backend that builds a live JuMP + MathOptVRP model in Julia and solves
it with Vroom.jl.

This backend handles models that contain:
  - VRPConstraint(variables, Partition(n, k))  — from m.constraint_in_set()
  - VRPObjective with OpSumDistances terms     — from jp.minimize(sum(...))

It does NOT handle standard JuMPy constraints or objectives — those go through
the juliacall backend (HiGHS).  A model with both VRP and standard constraints
is not supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.backend import Backend
from jumpy.vrp import (
    OpSumDistances,
    Partition,
    VRPObjective,
    _SumOfDistances,
)
from jumpy.expressions import Variable


class VroomBackend(Backend):
    """
    Calls Julia's Vroom.jl solver through juliacall.

    On first use, installs and loads:
        MathOptVRP, Vroom, JuMP
    """

    def __init__(self):
        self._jl = None

    # ── Julia initialisation ──────────────────────────────────────────────────

    def _init_julia(self):
        if self._jl is not None:
            return
        try:
            from juliacall import Main as jl
        except ImportError:
            raise ImportError(
                "juliacall is not installed.\n"
                "Install with: pip install jumpy[juliacall]"
            ) from None

        jl.seval("using Pkg")
        for pkg in ["JuMP", "MathOptVRP", "Vroom"]:
            jl.seval(f"""
                if !haskey(Pkg.project().dependencies, "{pkg}")
                    Pkg.add("{pkg}")
                end
            """)
        jl.seval("import JuMP")
        jl.seval("import MathOptVRP")
        jl.seval("import Vroom")
        self._jl = jl

    # ── Main entry point ──────────────────────────────────────────────────────

    def optimize(self, model: Model) -> list[float]:
        self._init_julia()
        jl = self._jl

        # --- validate model shape ---
        if not model._vrp_constraints:
            raise ValueError(
                "VroomBackend requires at least one constraint_in_set(variables, Partition(...))"
            )
        if not isinstance(model._objective, VRPObjective):
            raise ValueError(
                "VroomBackend requires a VRP objective built from op_sum_distances. "
                "Use jp.minimize(sum(op_sum_distances(...) for ...))"
            )

        # --- find the Partition constraint ---
        partition_con = None
        for con in model._vrp_constraints:
            if isinstance(con.set, Partition):
                partition_con = con
                break
        if partition_con is None:
            raise ValueError("No Partition constraint found in model")

        part = partition_con.set
        n = part.n_clients
        k = part.n_trucks

        # --- extract the distance matrix and depot from the objective ---
        obj_expr = model._objective.expr
        terms = (
            obj_expr.terms
            if isinstance(obj_expr, _SumOfDistances)
            else [obj_expr]
        )
        if len(terms) != k:
            raise ValueError(
                f"Objective has {len(terms)} op_sum_distances terms "
                f"but Partition has {k} trucks"
            )

        # all terms must share the same matrix and depot
        M     = terms[0].matrix
        depot = _extract_depot(terms[0])
        for t in terms[1:]:
            if t.matrix != M:
                raise ValueError("All op_sum_distances terms must use the same matrix")
            if _extract_depot(t) != depot:
                raise ValueError("All op_sum_distances terms must use the same depot")

        # matrix dimension check
        N = len(M)
        assert all(len(row) == N for row in M), "Distance matrix must be square"
        assert N == n + 1, f"Matrix must be (n_clients+1) x (n_clients+1), got {N}x{N}"

        # --- build the JuMP model in Julia ---
        jl_model = jl.JuMP.Model(jl.Vroom.Optimizer)
        jl.JuMP.set_silent(jl_model)

        # @variable(jl_model, nodes[1:n, 1:k] in MathOptVRP.Partition(n, k))
        partition_set = jl.MathOptVRP.Partition(n, k)
        jl.seval("""
            function _vroom_add_partition_vars(model, n, k, set)
                JuMP.@variable(model, nodes[1:n, 1:k] in set)
                return nodes
            end
        """)
        jl_nodes = jl._vroom_add_partition_vars(jl_model, n, k, partition_set)

        # build the distance matrix as a Julia matrix
        jl_M = jl.seval("(rows...) -> hcat(rows...)'")(
            *[jl.seval("collect")([int(M[i][j]) for j in range(N)])
              for i in range(N)]
        )

        # figure out which Python variable indices correspond to which truck column
        # The VariableVector is flat: variables[t * n + i] = truck t, position i
        # (column-major, matching JuMP's Partition layout)
        var_block = partition_con.variables  # VariableVector

        # map each op_sum_distances sequence to the Julia column index
        # The sequence is [depot_int, Var, Var, ..., Var, depot_int]
        # We identify which truck column each term corresponds to by the
        # variable indices in the sequence
        col_assignments = _assign_columns(terms, var_block, n, k)

        # @objective(jl_model, Min, sum(
        #     MathOptVRP.op_sum_distances(M, vcat(depot, nodes[:,i], depot))
        #     for i in 1:k
        # ))
        jl.seval("""
            function _vroom_set_objective(model, M, nodes, depot, n_trucks)
                JuMP.@objective(model, Min,
                    sum(
                        MathOptVRP.op_sum_distances(
                            M,
                            vcat(depot, nodes[:, i], depot)
                        )
                        for i in 1:n_trucks
                    )
                )
            end
        """)
        jl._vroom_set_objective(jl_model, jl_M, jl_nodes, depot, k)

        # --- solve ---
        jl.JuMP.optimize_b(jl_model)

        status = jl.JuMP.termination_status(jl_model)
        if str(status) not in ("OPTIMAL", "LOCALLY_SOLVED"):
            raise RuntimeError(f"Vroom did not find a solution: status = {status}")

        # --- extract routes ---
        # Vroom stores routes per partition column on the inner optimizer
        inner = jl.JuMP.unsafe_backend(jl_model)
        jl_routes = inner.routes   # Vector{Vector{Int}} in Julia
        routes = [
            [int(c) for c in jl_routes[i+1]]
            for i in range(int(jl.length(jl_routes)))
        ]
        model._routes = routes

        # return a flat solution vector so m.value(x) works
        # variables in the partition block get their assigned customer value
        # (0 if the slot is not used — Vroom may leave some slots empty)
        sol = [0.0] * model._num_vars
        for t, route in enumerate(routes):
            for pos, cust in enumerate(route):
                flat_idx = t * n + pos
                if flat_idx < len(var_block):
                    var = var_block[flat_idx]
                    sol[var.index] = float(cust)

        return sol


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_depot(term: OpSumDistances) -> int:
    """Get the depot index from an OpSumDistances sequence."""
    seq = term.sequence
    if not seq:
        raise ValueError("op_sum_distances sequence is empty")
    first = seq[0]
    if not isinstance(first, (int, float)):
        raise ValueError(
            f"First element of op_sum_distances sequence must be the depot "
            f"(an integer), got {type(first).__name__}"
        )
    return int(first)


def _assign_columns(terms, var_block, n, k):
    """
    For each op_sum_distances term, determine which Julia column (1-based)
    it corresponds to by inspecting which variable indices appear in the
    sequence and matching them to the column-major layout of the partition.
    """
    assignments = []
    for term in terms:
        seq = term.sequence
        # find the first Variable in the sequence
        for item in seq:
            if isinstance(item, Variable):
                # flat index in var_block: col = flat_idx // n  (0-based)
                local_idx = item.index - var_block[0].index
                col_0based = local_idx // n
                assignments.append(col_0based + 1)  # Julia is 1-based
                break
        else:
            raise ValueError(
                "op_sum_distances sequence contains no variables — "
                "cannot determine truck column"
            )
    return assignments