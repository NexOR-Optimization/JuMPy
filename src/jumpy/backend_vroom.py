"""
backend_vroom.py
================
JuMPy backend that builds a live JuMP + MathOptVRP model in Julia and solves
it with Vroom.jl.

This backend handles the VRP shape of a (generic) JuMPy Model:
  - one constraint_in_set(variables, Partition(n, k)) constraint
  - an objective that is a sum of op_sum_distances terms, one per truck

The shape is read and validated by the query helpers in jumpy.vrp
(find_partition / objective_terms). Standard JuMPy constraints and objectives
are not supported by Vroom — use the juliacall backend for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.backend import Backend
from jumpy.vrp import MATHOPTVRP_URL, find_partition, objective_terms


class VroomBackend(Backend):
    """
    Calls Julia's Vroom.jl solver through juliacall.

    On first use, installs and loads:
        JuMP, MathOptVRP, Vroom
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

        from jumpy.bridge_juliacall import ensure_package
        ensure_package(jl, "JuMP")
        ensure_package(jl, "MathOptVRP", url=MATHOPTVRP_URL)
        ensure_package(jl, "Vroom")
        self._jl = jl

    # ── Main entry point ──────────────────────────────────────────────────────

    def optimize(self, model: Model) -> list[float]:
        # --- read the VRP shape off the model (validates it too) ---
        partition_con = find_partition(model)
        terms = objective_terms(model, partition_con)

        part = partition_con.set_
        n, k = part.n_clients, part.n_trucks
        depot = _single_depot(terms)
        matrix = terms[0].matrix

        if model._objective.sense != "min":
            raise ValueError("Vroom only supports minimization objectives")
        N = len(matrix)
        if N != n + 1:
            raise ValueError(
                f"Vroom expects an (n_clients+1) x (n_clients+1) matrix, got {N}x{N}"
            )
        if model._constraint_groups or model._individual_constraints:
            raise ValueError(
                "VroomBackend only handles VRP models; standard constraints "
                "are not supported (use backend='juliacall')"
            )

        self._init_julia()
        jl = self._jl

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
            *[jl.seval("collect")([int(matrix[i][j]) for j in range(N)])
              for i in range(N)]
        )

        # @objective(jl_model, Min, sum(
        #     MathOptVRP.op_sum_distances(M, vcat(depot, nodes[:,i], depot))
        #     for i in 1:k
        # ))
        # objective_terms() guarantees term t visits partition column t, so the
        # Julia columns line up with the Python objective terms.
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
        # depot is already a 1-based location id (Julia matrix convention)
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
        var_block = partition_con.variables
        sol = [0.0] * model._num_vars
        for t, route in enumerate(routes):
            for pos, cust in enumerate(route):
                flat_idx = t * n + pos
                if flat_idx < len(var_block):
                    var = var_block[flat_idx]
                    sol[var.index] = float(cust)

        return sol


# ── helpers ───────────────────────────────────────────────────────────────────

def _single_depot(terms) -> int:
    """
    The one depot shared by every term's start and end.

    The Vroom wrapper builds each truck's route as vcat(depot, column, depot),
    so per-truck or asymmetric start/end depots are not supported here.
    """
    depots = {d for term in terms for d in (term.start, term.end)}
    if len(depots) != 1:
        raise ValueError(
            "VroomBackend requires all op_sum_distances terms to start and "
            f"end at the same depot, found depots {sorted(depots)}"
        )
    return depots.pop()
