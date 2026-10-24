"""
The Model class: top-level API for building optimization models in JuMPy.

The model is built eagerly: every call performs the corresponding JuMP call
through the backend's ops object (juliacall or the compiled library).
optimize() calls JuMP.optimize!; value() queries native expressions directly.
"""

from __future__ import annotations

import math

from jumpy.expressions import (
    _native,
    Constraint,
    Node,
    Objective,
    Parameter,
    Variable,
    VariableVector,
)


def minimize(func: Node) -> Objective:
    return Objective("min", func)


def maximize(func: Node) -> Objective:
    return Objective("max", func)


class Model:
    """
    A JuMPy optimization model.

    Example:
        m = jp.Model()
        x = m.variables(100, lower=0)

        i = m.iterator(range(99))
        m.constraint_group(x[i] + x[i + 1] <= 10)

        m.objective = jp.minimize(x[0] + x[1])
        m.optimize()
    """

    def __init__(self, ops):
        """Internal implementation; public models select ops via their module."""
        self._ops = ops
        self._objective: Objective | None = None

    def close(self) -> None:
        """Release the backend model. The model must not be used afterwards."""
        # getattr: __del__ may run when __init__ failed before setting _ops
        ops = getattr(self, "_ops", None)
        if ops is not None:
            ops.free()
            self._ops = None

    def __del__(self):
        self.close()

    def _require_open(self):
        if self._ops is None:
            raise RuntimeError("The model has been closed")

    # -- Variables -------------------------------------------------------------

    def variables(
        self,
        count: int,
        *,
        lower: float | None = None,
        upper: float | None = None,
        name: str | None = None,
        binary: bool = False,
        integer: bool = False,
    ) -> VariableVector:
        """Add a native Julia array of JuMP variables and their bounds."""
        if not isinstance(count, int):
            raise TypeError("Variable count must be an integer")
        if count < 0:
            raise ValueError("Variable count must be nonnegative")
        variables = VariableVector(self._ops, self._ops.add_variables(count), count, name)
        # Bounds and integrality are VariableIndex-in-set constraints, as in MOI.
        # Reuse each set across the array instead of constructing it per variable.
        moi = self._ops.MOI
        sets = []
        if lower is not None:
            sets.append(moi.GreaterThan(float(lower)))
        if upper is not None:
            sets.append(moi.LessThan(float(upper)))
        if binary:
            sets.append(moi.ZeroOne())
        elif integer:
            sets.append(moi.Integer())
        if sets:
            for variable in variables:
                for set_ in sets:
                    self._ops.add_constraint(variable.ref, set_)
        return variables

    def variable(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        name: str | None = None,
        binary: bool = False,
        integer: bool = False,
    ) -> Variable:
        """Add a single decision variable."""
        return self.variables(
            1, lower=lower, upper=upper, name=name, binary=binary, integer=integer,
        )[0]

    # -- Template data -----------------------------------------------------------

    def iterator(self, values) -> Node:
        """
        An index set for constraint groups (a GenOpt iterator).

        Used in expressions, it is a symbolic placeholder that GenOpt
        expands over its values when the group constraint is added.
        """
        return Node(self._ops, self._ops.iterator(list(values)))

    def parameter(self, values, name: str | None = None) -> Parameter:
        """A vector of constant data, symbolically indexable in templates."""
        return Parameter(self._ops, values, name)

    # -- Constraints -----------------------------------------------------------

    def constraint(self, func, set_=None) -> None:
        """Add ``constraint(x <= 1)`` or ``constraint(x, jp.MOI.LessThan(1.0))``.

        Comparison syntax forwards the same function and native set to the
        explicit form. Julia handles simplification and constant normalization.
        """
        self._require_open()
        if set_ is None:
            if not isinstance(func, Constraint):
                raise TypeError("Expected a comparison constraint, or a function and an MOI set")
            return self.constraint(func.func, func.set)
        self._ops.add_constraint(_native(self._ops, func), set_)

    def constraint_group(self, con: Constraint) -> None:
        """
        Add a constraint group: one constraint per combination of the
        values of the iterators appearing in the template.

        Example:
            i = m.iterator(range(99))
            m.constraint_group(x[i] + x[i + 1] <= 10)
        """
        return self.constraint(con)

    # -- Objective -------------------------------------------------------------

    def lazy_sum(self, template) -> Node:
        """Sum a GenOpt template without expanding its symbolic iterators.

        For example, ``model.lazy_sum(cost[i] * x[i])`` retains the repeated
        expression's structure instead of consuming a Python iterable.
        """
        self._require_open()
        return Node(
            self._ops,
            self._ops.apply("lazy_sum", [_native(self._ops, template)]),
        )

    @property
    def objective(self) -> Objective | None:
        return self._objective

    @objective.setter
    def objective(self, obj: Objective) -> None:
        self._objective = obj
        self._ops.set_objective(obj.sense, _native(self._ops, obj.func))

    # -- Solve -----------------------------------------------------------------

    def set_time_limit_sec(self, seconds: float | None) -> None:
        """Set a positive finite solve time limit in seconds, or clear it with None."""
        if self._ops is None:
            raise RuntimeError("The model has been closed")
        if seconds is not None:
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("The time limit must be positive and finite, or None")
        self._ops.set_time_limit_sec(seconds)

    def unset_time_limit_sec(self) -> None:
        """Clear the model's solve time limit."""
        self.set_time_limit_sec(None)

    def optimize(self) -> int:
        """Solve and return the native MOI termination status code."""
        self._require_open()
        return self._ops.optimize()

    def termination_status(self) -> int:
        """Return the native MOI termination status code reported by JuMP."""
        self._require_open()
        return self._ops.termination_status()

    def has_values(self) -> bool:
        """Whether JuMP reports an available primal solution."""
        self._require_open()
        return self._ops.has_values()

    def has_duals(self) -> bool:
        """Whether JuMP reports an available dual solution (normally not for MIPs)."""
        self._require_open()
        return self._ops.has_duals()

    def objective_value(self) -> float:
        """Query the solver's current objective value."""
        self._require_open()
        return self._ops.objective_value()

    def value(self, expr: Node) -> float:
        """Query JuMP for the value of a native variable or scalar expression."""
        self._require_open()
        return self._ops.value(_native(self._ops, expr))
