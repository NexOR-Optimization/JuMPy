"""
VRP extension for JuMPy (MathOptVRP).

There is no separate "VRP model": VRP pieces live in the one generic jp.Model
like everything else, through the two extension points defined in
jumpy.expressions:

  * VectorSet      -> Partition        m.constraint_in_set(nodes, Partition(n, k))
  * SolverFunction -> OpSumDistances   jp.minimize(sum(op_sum_distances(...) ...))

OpSumDistances is a real Expr, so per-vehicle cost terms combine with plain
`+` / `sum(...)` and flow through minimize(), the juliacall bridge and the MOF
writer like any other expression — the core has no VRP special cases.

Adding a new VRP set or function:
  1. subclass VectorSet or SolverFunction here,
  2. implement `to_moi` (and `setup_julia` if it needs extra Julia packages
     or MathOptFormat writer methods).
Nothing in model.py / bridge_juliacall.py needs to change; write_mof picks the
new pieces up through the same protocol.

The module also provides query helpers (find_partition, objective_terms) that
VRP-shaped consumers (the Vroom backend) use to read and validate the relevant
parts of a Model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model, SetConstraint

from jumpy.expressions import (
    BinaryOp,
    Constant,
    SolverFunction,
    Variable,
    VectorSet,
)

MATHOPTVRP_URL = "https://github.com/NexOR-Optimization/MathOptVRP.jl"


def _normalize_matrix(matrix) -> list[list[int]]:
    """
    Coerce a distance matrix into list[list[int]] (matrix[i][j]).

    Accepts either an already-nested matrix or a flat, row-major sequence
    (matrix[i * n + j]) — the latter is what most distance-matrix producers
    (numpy .flatten(), routing APIs, etc.) hand back.
    """
    rows = list(matrix)
    if rows and not hasattr(rows[0], "__iter__"):
        n = math.isqrt(len(rows))
        if n * n != len(rows):
            raise ValueError(
                f"Flat distance matrix has {len(rows)} entries, "
                "which is not a perfect square"
            )
        return [list(rows[i * n:(i + 1) * n]) for i in range(n)]
    return [list(row) for row in rows]


# ── Set types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Partition(VectorSet):
    """
    MathOptVRP.Partition(n_clients, n_trucks).

    Declares that the associated variables form a partition of n_clients
    customers across n_trucks vehicles. The variables are laid out column-
    major: column k (0-based) holds the route slots for truck k.
    """
    n_clients: int
    n_trucks:  int

    @property
    def dimension(self) -> int:
        return self.n_clients * self.n_trucks

    @classmethod
    def setup_julia(cls, jl) -> None:
        from jumpy.bridge_juliacall import ensure_package
        ensure_package(jl, "MathOptVRP", url=MATHOPTVRP_URL)
        jl.seval(
            'MOI.FileFormats.MOF.head_name(::Type{MathOptVRP.Partition}) = "Partition"'
        )

    def to_moi(self, jl):
        return jl.MathOptVRP.Partition(self.n_clients, self.n_trucks)

    def variable_names(self, variables) -> list[str]:
        # Match Julia's `@variable(m, nodes[1:n, 1:k] in Partition)` naming:
        # 2-D, 1-based, column-major (flat slot t*n + i  ->  base[i+1, t+1]).
        base = variables.name or "nodes"
        return [
            f"{base}[{i + 1},{t + 1}]"
            for t in range(self.n_trucks)
            for i in range(self.n_clients)
        ]


# ── Solver functions ──────────────────────────────────────────────────────────

class OpSumDistances(SolverFunction):
    """
    Symbolic MathOptVRP.op_sum_distances(M, sequence): the routing cost of one
    vehicle along `sequence`.

    Parameters
    ----------
    matrix   : square distance matrix (nested, or flat row-major)
    sequence : list — [start_depot:int, Variable, ..., Variable, end_depot:int].
               Depot / fixed-stop ids are 1-based location indices into the
               matrix (Julia convention) and are passed through unchanged.
    """

    head = "sum_distances"

    def __init__(self, matrix, sequence: list):
        self.matrix = _normalize_matrix(matrix)
        self.sequence = list(sequence)
        if len(self.sequence) < 2:
            raise ValueError(
                "op_sum_distances sequence needs at least a start and an end stop"
            )
        for label, stop in (("start", self.sequence[0]), ("end", self.sequence[-1])):
            if not isinstance(stop, (int, float)):
                raise ValueError(
                    f"The {label} of an op_sum_distances sequence must be a depot "
                    f"index (int), got {type(stop).__name__}"
                )

    @property
    def start(self) -> int:
        """Start depot index."""
        return int(self.sequence[0])

    @property
    def end(self) -> int:
        """End depot index."""
        return int(self.sequence[-1])

    def __repr__(self) -> str:
        n = len(self.matrix)
        return f"sum_distances({n}x{n}, {self.sequence})"

    @classmethod
    def setup_julia(cls, jl) -> None:
        # MathOptFormat writer methods for the two leaf types this function
        # emits: the distance matrix and the stop sequence.
        jl.seval("""
            # A distance matrix appears as a leaf of the `:sum_distances`
            # nonlinear node. Serialize it as a self-describing "matrix" object.
            function MOI.FileFormats.MOF._convert_nonlinear_to_mof(
                value::AbstractMatrix{<:Real},
                ::Vector{Any},
                ::Dict{MOI.VariableIndex,String},
            )
                return (
                    type = "matrix",
                    size = [size(value, 1), size(value, 2)],
                    values = [value[i, :] for i in 1:size(value, 1)],
                )
            end

            # The stop sequence `[depot; vars...; depot]` reaches the writer as
            # a `VectorAffineFunction`. Emit it as a "sequence" of simplified
            # leaves:
            #   * a bare number for the depot / constant entries,
            #   * a bare variable-name string for a plain unit variable,
            #   * a nested affine subtree otherwise.
            # Keeping the simple cases as leaves lets the reader rebuild exactly
            # the `[depot; vars...; depot]` vector that the Vroom wrapper expects.
            function MOI.FileFormats.MOF._convert_nonlinear_to_mof(
                f::MOI.VectorAffineFunction{T},
                node_list::Vector{Any},
                name_map::Dict{MOI.VariableIndex,String},
            ) where {T<:Real}
                n = MOI.output_dimension(f)
                rows = [MOI.ScalarAffineTerm{T}[] for _ in 1:n]

                for vt in f.terms
                    push!(rows[vt.output_index], vt.scalar_term)
                end

                args = Any[]
                for i in 1:n
                    terms_i, const_i = rows[i], f.constants[i]
                    if isempty(terms_i)
                        push!(args, const_i)
                    elseif length(terms_i) == 1 && isone(terms_i[1].coefficient) && iszero(const_i)
                        push!(args, name_map[terms_i[1].variable])
                    else
                        s = MOI.ScalarAffineFunction{T}(terms_i, const_i)
                        push!(args, MOI.FileFormats.MOF._convert_nonlinear_to_mof(s, node_list, name_map))
                    end
                end
                push!(node_list, (type = "sequence", args = args))
                return (type = "node", index = length(node_list))
            end
        """)

    def to_moi(self, ctx):
        # Location ids are already 1-based (Julia matrix convention) and pass
        # through unchanged — only variable indices get the 0->1 based shift.
        return ctx.nonlinear(
            self.head,
            ctx.matrix(self.matrix),
            ctx.vector_affine(self.sequence),
        )


def op_sum_distances(matrix, sequence) -> OpSumDistances:
    """
    Routing cost for one truck along `sequence`.

    Parameters
    ----------
    matrix   : square distance matrix
    sequence : list — [depot_int, Variable, ..., Variable, depot_int], where
               depot ids are 1-based location indices into the matrix

    Returns an OpSumDistances expression to be used inside jp.minimize(...).
    """
    return OpSumDistances(matrix, sequence)


# ── Model query helpers ───────────────────────────────────────────────────────
#
# Used by consumers that only handle VRP-shaped models (the Vroom backend, the
# JSON exporter) to read the relevant pieces out of the one generic Model.

def find_partition(model: Model) -> SetConstraint:
    """Return the model's single Partition set constraint, validating there is one."""
    partition_cons = [
        c for c in model._set_constraints if isinstance(c.set_, Partition)
    ]
    if len(partition_cons) != 1:
        raise ValueError(
            f"Expected exactly one Partition constraint, found {len(partition_cons)}. "
            "Add one with m.constraint_in_set(variables, Partition(n, k))."
        )
    return partition_cons[0]


def _iter_distance_terms(expr):
    """Yield the OpSumDistances leaves of a `+`-tree objective expression."""
    if isinstance(expr, OpSumDistances):
        yield expr
    elif isinstance(expr, BinaryOp) and expr.op == "+":
        yield from _iter_distance_terms(expr.left)
        yield from _iter_distance_terms(expr.right)
    elif isinstance(expr, Constant) and expr.value == 0.0:
        return  # `sum(...)` starts from 0
    else:
        raise ValueError(
            "VRP objective must be a sum of op_sum_distances terms, "
            f"found {type(expr).__name__}"
        )


def objective_terms(model: Model, partition_con: SetConstraint) -> list[OpSumDistances]:
    """
    Return the objective's OpSumDistances terms, one per truck, ordered so
    that terms[t] is the term visiting truck t's partition column.

    Validates the whole VRP objective shape: term count == n_trucks, all terms
    share one square distance matrix, and each term's interior stops are
    exactly one truck's column of the partition block, in order.
    """
    if model._objective is None:
        raise ValueError(
            "VRP models need an objective. "
            "Use jp.minimize(sum(op_sum_distances(...) for ...))"
        )
    terms = list(_iter_distance_terms(model._objective.expr))

    partition: Partition = partition_con.set_
    n, k = partition.n_clients, partition.n_trucks
    if len(terms) != k:
        raise ValueError(
            f"Objective has {len(terms)} op_sum_distances terms "
            f"but Partition has {k} trucks"
        )

    matrix = terms[0].matrix
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Distance matrix must be square")
    if any(t.matrix != matrix for t in terms[1:]):
        raise ValueError("All op_sum_distances terms must use the same matrix")

    # Map each term to its truck by matching its interior stops against the
    # column-major partition layout (truck t owns variables[t*n : (t+1)*n]).
    base = partition_con.variables._variables[0].index
    ordered: list[OpSumDistances | None] = [None] * k
    for term in terms:
        stops = term.sequence[1:-1]
        if len(stops) != n or not all(isinstance(s, Variable) for s in stops):
            raise ValueError(
                "Each op_sum_distances sequence must visit exactly the "
                f"{n} route-slot variables of one truck between its depots"
            )
        truck = (stops[0].index - base) // n
        expected = [base + truck * n + i for i in range(n)]
        if truck not in range(k) or [s.index for s in stops] != expected:
            raise ValueError(
                "op_sum_distances stops must be one truck's partition column "
                "in order (variables[t*n], ..., variables[t*n + n-1])"
            )
        if ordered[truck] is not None:
            raise ValueError(
                f"Two op_sum_distances terms visit truck {truck}'s variables"
            )
        ordered[truck] = term
    return ordered
