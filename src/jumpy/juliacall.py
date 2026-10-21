"""JuliaCall backend: ``import jumpy.juliacall as jp``.

Julia is initialized lazily, when creating a model or accessing ``jp.MOI``.
"""

from jumpy import *  # Re-export the common modeling API.
from jumpy import __all__ as _common_all
from jumpy import bridge_juliacall as _bridge
from jumpy.model import Model as _Model

__all__ = [*_common_all, "Model", "MOI"]


class Model(_Model):
    """A model using HiGHS through JuliaCall and MOI bridges."""

    def __init__(self):
        super().__init__(_bridge.JuliaCallOps())


def __getattr__(name):
    if name == "MOI":
        return _bridge._julia().MOI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
