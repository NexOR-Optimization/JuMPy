"""
The Model class: top-level API for building optimization models in JuMPy.

The model is built eagerly: every call performs the corresponding JuMP call
through the backend's ops object (juliacall or the compiled library).
optimize() calls JuMP.optimize! and retrieves the solution.
"""

from __future__ import annotations

from jumpy.expressions import (
    _native,
    Constraint,
    Node,
    Objective,
    Parameter,
    Variable,
    VariableVector,
)

# MOI.OPTIMAL in MOI.TerminationStatusCode.
OPTIMAL = 1


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
        self._num_vars = 0
        self._objective: Objective | None = None
        self._solution: list[float] | None = None

    def close(self) -> None:
        """Release the backend model. The model must not be used afterwards."""
        # getattr: __del__ may run when __init__ failed before setting _ops
        ops = getattr(self, "_ops", None)
        if ops is not None:
            ops.free()
            self._ops = None

    def __del__(self):
        self.close()

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
        """Add a block of native JuMP decision variables and their bounds."""
        start = self._ops.add_variables(count)
        # Bounds and integrality are VariableIndex-in-set constraints, as in MOI.
        # Reuse each set across the block instead of constructing it per variable.
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
        for k in range(count):
            for set_ in sets:
                self._ops.add_constraint(self._ops.variable(start + k), set_)
        self._num_vars += count
        return VariableVector(self._ops, start, count, name)

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
        if self._ops is None:
            raise RuntimeError("The model has been closed")
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

    @property
    def objective(self) -> Objective | None:
        return self._objective

    @objective.setter
    def objective(self, obj: Objective) -> None:
        self._objective = obj
        self._ops.set_objective(obj.sense, _native(self._ops, obj.func))

    # -- Solve -----------------------------------------------------------------

    def optimize(self) -> None:
        """JuMP.optimize!, then retrieve the solution."""
        status = self._ops.optimize()
        if status != OPTIMAL:
            raise RuntimeError(
                f"Solve did not reach OPTIMAL (termination status {status})"
            )
        self._solution = self._ops.get_values(self._num_vars)

    def value(self, var: Variable) -> float:
        """Get the solved value of a variable."""
        if self._solution is None:
            raise RuntimeError("Model has not been solved yet. Call optimize() first.")
        return self._solution[var.index]
