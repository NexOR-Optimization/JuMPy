"""
VRP-specific modeling objects for JuMPy.

These are intentionally opaque data containers, they carry the information
that MathOptVRP and the solver need, but have no mathematical semantics in JuMPy
itself. They are passed through unchanged to the Vroom backend, which builds
the live Julia model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Set types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Partition:
    """
    MathOptVRP.Partition(n_clients, n_trucks).

    Declares that the associated variables form a partition of n_clients
    customers across n_trucks vehicles.  The variables are laid out column-
    major: column k (0-based) holds the route slots for truck k.
    """
    n_clients: int
    n_trucks:  int

    @property
    def dimension(self):
        return self.n_clients * self.n_trucks

# ── Expression node ───────────────────────────────────────────────────────────

class OpSumDistances:
    """
    Symbolic representation of MathOptVRP.op_sum_distances(M, sequence).

    not a JuMPy Expr subclass — it is never walked by the affine/nonlinear
    expression parser.  It is stored directly as the model objective's
    inner value and handed to the Vroom backend as opaque data.

    Parameters
    ----------
    matrix   : list[list[int]]  — square integer distance matrix
    sequence : list             — mix of int (depot) and Variable (customers) (vehicle path)
    """

    def __init__(self, matrix: list[list[int]], sequence: list):
        self.matrix   = matrix
        self.sequence = sequence

    def __add__(self, other):
        if isinstance(other, OpSumDistances):
            return _SumOfDistances([self, other])
        if isinstance(other, _SumOfDistances):
            return _SumOfDistances([self] + other.terms)
        return NotImplemented

    def __radd__(self, other):
        # support sum([op1, op2, ...]) which starts with 0 + op1 (necessary for objective def)
        if other == 0:
            return self
        return NotImplemented


class _SumOfDistances:
    """Internal: sum of multiple OpSumDistances (one per truck)."""

    def __init__(self, terms: list[OpSumDistances]):
        self.terms = terms

#    def __add__(self, other):
        #if isinstance(other, OpSumDistances):
            #return _SumOfDistances(self.terms + [other])
        #if isinstance(other, _SumOfDistances):
            #return _SumOfDistances(self.terms + other.terms)
        #return NotImplemented

def op_sum_distances(matrix, sequence):
    """
    Routing cost for one truck along `sequence`.

    Parameters
    ----------
    matrix   : list[list[int]]   — (n+1)x(n+1) distance matrix
    sequence : list              — [depot_int, Variable, ..., Variable, depot_int]

    Returns an OpSumDistances node to be used inside jp.minimize(...).
    """
    return OpSumDistances(matrix, list(sequence))


# ── Function-in-set constraint record ────────────────────────────────────────

@dataclass
class VRPConstraint:
    """
    Records a `variables in set` constraint for the Vroom backend.

    variables : VariableVector — the flat variable block
    set       : Partition      — the set they are constrained to
    """
    variables: Any   # VariableVector
    set:       Any   # Partition (or future set types)


# ── VRP Objective wrapper ─────────────────────────────────────────────────────

@dataclass
class VRPObjective:
    """
    Wraps a sum-of-distances objective so Model._objective can hold it.
    The sense is always considered 'min'.
    """
    sense: str                          # always "min"
    expr:  OpSumDistances | _SumOfDistances