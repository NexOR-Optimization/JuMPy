"""JuliaCall backend: ``import jumpy.juliacall as jp``.

Julia is initialized lazily, when creating a model or accessing ``jp.MOI``.
"""

from jumpy import *  # Re-export the common modeling API.
from jumpy import __all__ as _common_all
from jumpy import bridge_juliacall as _bridge
from jumpy.model import Model as _Model

__all__ = [*_common_all, "Model", "MOI"]


class Model(_Model):
    """A JuliaCall model with MOI bridges, using HiGHS by default.

    ``optimizer`` may be a native Julia MOI optimizer instance, not a factory.
    """

    def __init__(self, optimizer=None):
        ops = _bridge.JuliaCallOps() if optimizer is None else _bridge.JuliaCallOps(optimizer)
        super().__init__(ops)


def __getattr__(name):
    if name == "MOI":
        return _bridge._julia().MOI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
