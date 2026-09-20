"""Native MOI constructors in the compiled image, with checked opaque handles.

These wrappers do not define MOI sets or store copies of their fields. Objects
live in the library's Julia registry until closed; JuliaCall may borrow them
only when it is attached to that same, untrimmed image.
"""

from __future__ import annotations

import ctypes

from jumpy import backend


class MOIObject:
    """An owned reference to a native MOI value; use as a context manager."""

    def __init__(self, owner, handle):
        if not handle:
            raise RuntimeError("Failed to construct native MOI object")
        self._owner = owner
        self._handle = int(handle)

    @property
    def handle(self):
        if not self._handle:
            raise RuntimeError("Native MOI object has been closed")
        self._owner._check_live()
        return self._handle

    def close(self):
        handle = getattr(self, "_handle", 0)
        if handle:
            if not backend._SHUTTING_DOWN:
                if self._owner._lib.jumpy_moi_release(handle) != 0:
                    raise RuntimeError("Failed to release native MOI object")
            self._handle = 0

    def __enter__(self):
        self.handle  # fail immediately for a closed object
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # explicit close reports errors; finalizers must not raise

    def to_julia(self):
        """Borrow the actual object into JuliaCall's GC, without serialization."""
        handle = self.handle
        jl = self._owner._julia()
        borrow = jl.seval("h -> JuMPyHiGHS.JuMPyMOIABI.get_object(UInt64(h))")
        return borrow(handle)


class MOIConstructors:
    """Low-level constructors backed by one compiled library's object registry.

    Scalar senses are ``<=``, ``>=``, ``==``, ``binary``, and ``integer``.
    Variable indices here are MOI's one-based indices, not Python column indices.
    This API does not add a public ``Model.constraint(func, set)`` overload.
    """

    def __init__(self, lib=None):
        self._check_live()
        self._lib = backend._load_lib() if lib is None else lib

    @staticmethod
    def _check_live():
        if backend._SHUTTING_DOWN:
            raise RuntimeError("The Julia runtime is shutting down")

    def scalar_set(self, sense, rhs=0.0):
        self._check_live()
        return MOIObject(self, self._lib.jumpy_moi_scalar_set(
            backend._SENSE_CODES[sense], float(rhs),
        ))

    def vector_set(self, sense, dimension):
        self._check_live()
        return MOIObject(self, self._lib.jumpy_moi_vector_set(
            backend._SENSE_CODES[sense], dimension,
        ))

    def constant(self, value):
        self._check_live()
        return MOIObject(self, self._lib.jumpy_moi_constant(float(value)))

    def variable(self, index):
        self._check_live()
        return MOIObject(self, self._lib.jumpy_moi_variable(index))

    def scalar_nonlinear(self, head, args):
        self._check_live()
        handles = []
        for arg in args:
            if not isinstance(arg, MOIObject):
                raise TypeError("Nonlinear arguments must be native MOI objects")
            if arg._owner._lib is not self._lib:
                raise ValueError("Nonlinear argument belongs to a different library")
            handles.append(arg.handle)
        argv = (ctypes.c_uint64 * len(handles))(*handles)
        return MOIObject(self, self._lib.jumpy_moi_scalar_nonlinear(
            head.encode(), argv, len(handles),
        ))

    def _julia(self):
        self._check_live()
        from jumpy.bridge_juliacall import _julia

        jl = _julia(shared_lib=self._lib)
        backend._check_shared_runtime(self._lib, jl)
        return jl

    def from_julia(self, value):
        """Root a native JuliaCall value in this same image's checked registry."""
        jl = self._julia()
        return MOIObject(self, jl.JuMPyHiGHS.JuMPyMOIABI.keep(value))
