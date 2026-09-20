"""
JuMPy: A Python interface to MathOptInterface via GenOpt.

Select a backend with ``import jumpy.highs as jp`` or
``import jumpy.juliacall as jp``. This module exports backend-neutral helpers.

Models are built eagerly: every operation performs the corresponding MOI
call, either in the compiled Julia library (juliac backend, no Julia
installation needed) or through juliacall.
"""

from jumpy.expressions import (
    Constraint,
    Node,
    Objective,
    Parameter,
    Variable,
    VariableVector,
)
from jumpy.expressions import sin, cos, exp, log, sqrt, abs as jp_abs
from jumpy.model import minimize, maximize

__all__ = [
    "Node",
    "Variable",
    "VariableVector",
    "Parameter",
    "Constraint",
    "Objective",
    "minimize",
    "maximize",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "jp_abs",
]
