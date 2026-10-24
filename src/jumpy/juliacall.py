"""JuliaCall backend: ``import jumpy.juliacall as jp``.

Julia is initialized lazily, when creating a model or accessing ``jp.MOI``.
"""

from jumpy import *  # Re-export the common modeling API.
from jumpy import __all__ as _common_all
from jumpy import bridge_juliacall as _bridge
from jumpy.model import Model as _Model

__all__ = [*_common_all, "Model", "direct_model", "MOI"]


class Model(_Model):
    """A cached JuMP model, optionally initialized with an optimizer factory.

    With no factory, the model has no optimizer. Native Julia constructors and
    Python callables are accepted; keyword arguments are forwarded to JuMP.
    """

    def __init__(self, optimizer=None, **kwargs):
        jl = _bridge._julia()
        native = jl.JuMPyModel.model(_bridge._optimizer_factory(optimizer), **kwargs)
        super().__init__(_bridge.JuliaCallOps(native))


def direct_model(optimizer, **kwargs):
    """Wrap ``JuMP.direct_model(optimizer)`` without adding caches or bridges."""
    jl = _bridge._julia()
    return _Model(_bridge.JuliaCallOps(jl.JuMP.direct_model(optimizer, **kwargs)))


def __getattr__(name):
    if name == "MOI":
        return _bridge._julia().MOI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
