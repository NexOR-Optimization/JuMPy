"""
JuMPy: A Python interface to MathOptInterface via GenOpt.

Select a backend with ``import jumpy.highs as jp`` or
``import jumpy.juliacall as jp``. This module exports backend-neutral helpers.

Models are built eagerly: every operation performs the corresponding Julia
call, either in the compiled Julia library (juliac backend, no Julia
installation needed) or through juliacall.
"""

# Compiled backends distributed separately, such as jumpy-opt-<name>, install
# the module jumpy/<name>.py and the bundle jumpy/_backends/<name> without an
# __init__.py. Extend the search path to these directories when they are not
# in this one, as for an editable installation of jumpy.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from jumpy.expressions import (
    Constraint,
    ConstraintRef,
    Node,
    Objective,
    Parameter,
    Variable,
    VariableArray,
    VariableVector,
    vcat,
)
from jumpy.expressions import sin, cos, exp, log, sqrt, abs as jp_abs
from jumpy.model import minimize, maximize

__all__ = [
    "Node",
    "Variable",
    "VariableArray",
    "VariableVector",
    "vcat",
    "Parameter",
    "Constraint",
    "ConstraintRef",
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
