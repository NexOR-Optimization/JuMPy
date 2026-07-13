"""
The Model class: top-level API for building optimization models in JuMPy.

The model is built eagerly: every call performs the corresponding MOI call
through the backend's ops object (juliacall or the compiled library).
optimize() is just MOI.optimize! plus solution retrieval.
"""

from __future__ import annotations

from jumpy.backend import get_ops
from jumpy.expressions import (
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

    def __init__(self, backend: str = "juliac"):
        """
        Create a new model.

        Args:
            backend: "juliac" (default, no Julia needed) or "juliacall"
                     (uses juliacall, installs Julia lazily if needed).
        """
        self._ops = get_ops(backend)
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
        """Add a block of decision variables (MOI.add_variables + bounds)."""
        start = self._ops.add_variables(count)
        # Bounds and integrality are VariableIndex-in-set constraints, as in MOI.
        for k in range(count):
            if lower is not None:
                self._ops.add_constraint(
                    self._ops.variable(start + k), ">=", float(lower),
                )
            if upper is not None:
                self._ops.add_constraint(
                    self._ops.variable(start + k), "<=", float(upper),
                )
            if binary:
                self._ops.add_constraint(self._ops.variable(start + k), "binary", 0.0)
            elif integer:
                self._ops.add_constraint(self._ops.variable(start + k), "integer", 0.0)
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
        return Node(self._ops, self._ops.iterator([float(v) for v in values]))

    def parameter(self, values, name: str | None = None) -> Parameter:
        """A vector of constant data, symbolically indexable in templates."""
        return Parameter(self._ops, values, name)

    # -- Constraints -----------------------------------------------------------

    def constraint(self, con: Constraint) -> None:
        """Add a single constraint (MOI.add_constraint)."""
        self._ops.add_constraint(con.func.moi, con.sense, 0.0)

    def constraint_group(self, con: Constraint) -> None:
        """
        Add a constraint group: one constraint per combination of the
        values of the iterators appearing in the template.

        Example:
            i = m.iterator(range(99))
            m.constraint_group(x[i] + x[i + 1] <= 10)
        """
        self._ops.add_constraint_group(con.func.moi, con.sense, con.func.linear)

    # -- Objective -------------------------------------------------------------

    @property
    def objective(self) -> Objective | None:
        return self._objective

    @objective.setter
    def objective(self, obj: Objective) -> None:
        self._objective = obj
        self._ops.set_objective(obj.sense, obj.func.moi)

    # -- Solve -----------------------------------------------------------------

    def optimize(self) -> None:
        """MOI.optimize!, then retrieve the solution."""
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
