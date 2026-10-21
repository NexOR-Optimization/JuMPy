"""Compiled HiGHS backend: ``import jumpy.highs as jp``.

Importing this module does not initialize Julia or require JuliaCall.
"""

from jumpy import *  # Re-export the common modeling API.
from jumpy import __all__ as _common_all
from jumpy import backend as _backend
from jumpy._highs_moi import MOI
from jumpy.model import Model as _Model

__all__ = [*_common_all, "Model", "MOI"]


class Model(_Model):
    """A model using the compiled HiGHS library."""

    def __init__(self):
        super().__init__(_backend.JuliacOps(_backend._load_lib()))
