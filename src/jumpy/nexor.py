"""Compiled NexOR backend: ``import jumpy.nexor as jp``.

Importing this module does not initialize Julia or contact the service.
"""

from jumpy import *
from jumpy import __all__ as _common_all
from jumpy import backend as _backend
from jumpy._moi import ScalarSets as _ScalarSets
from jumpy.model import Model as _Model

MOI = _ScalarSets("nexor")
__all__ = [*_common_all, "Model", "MOI"]


class Model(_Model):
    """A model using the compiled NexOR client. Select the remote solver with
    ``model.set_attribute("solver", "HiGHS")`` before optimizing."""

    def __init__(self):
        super().__init__(_backend.JuliacOps(_backend._load_lib("nexor"), MOI))
