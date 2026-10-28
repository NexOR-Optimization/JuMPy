"""Thin bindings to native MOI sets in the compiled backends.

Only ownership lives in Python. Set definitions and construction stay in Julia.
"""

from jumpy import backend


class NativeSet:
    """An owned native set; automatically released, or closed explicitly."""

    def __init__(self, lib, handle):
        if not handle:
            raise RuntimeError("Failed to construct the native MOI set")
        self._lib = lib
        self._handle = int(handle)

    @property
    def handle(self):
        if not self._handle:
            raise RuntimeError("The native MOI set has been closed")
        return self._handle

    def close(self):
        handle = getattr(self, "_handle", 0)
        if handle:
            if self._lib.jumpy_free_set(handle) != 0:
                raise RuntimeError("Failed to release the native MOI set")
            self._handle = 0

    def __enter__(self):
        self.handle
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _ScalarSets:
    """MOI constructor bindings for a compiled backend."""

    def __init__(self, solver="highs"):
        self._solver = solver

    def _new(self, constructor, *args):
        lib = backend._load_lib() if self._solver == "highs" else backend._load_lib(self._solver)
        return NativeSet(lib, getattr(lib, constructor)(*args))

    def LessThan(self, upper):
        """Construct a native MOI.LessThan{Float64}."""
        return self._new("jumpy_less_than", float(upper))

    def GreaterThan(self, lower):
        """Construct a native MOI.GreaterThan{Float64}."""
        return self._new("jumpy_greater_than", float(lower))

    def EqualTo(self, value):
        """Construct a native MOI.EqualTo{Float64}."""
        return self._new("jumpy_equal_to", float(value))

    def ZeroOne(self):
        """Construct a native MOI.ZeroOne."""
        return self._new("jumpy_zero_one")

    def Integer(self):
        """Construct a native MOI.Integer."""
        return self._new("jumpy_integer")

    def __getattr__(self, name):
        solver_name = "HiGHS" if self._solver == "highs" else "NexOR"
        raise AttributeError(
            f"MOI.{name} is not exposed by the compiled {solver_name} backend. "
            "Use jumpy.juliacall for other MOI constructors."
        )


MOI = _ScalarSets()
