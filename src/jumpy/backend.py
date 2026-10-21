"""
The compiled-library (juliac) JuMP operations.

An "ops" object maps each Julia call either to the compiled shared library
(JuliacOps below, via ctypes) or to Julia through juliacall
(jumpy.bridge_juliacall.JuliaCallOps). Models call the ops directly; the
two implementations expose the same methods.
"""

from __future__ import annotations

import ctypes
import math
import sys


# The Julia runtime can only be initialized once per process.
_LIB = None


def _default_paths():
    import os
    import platform

    soext = {"Linux": ".so", "Darwin": ".dylib", "Windows": ".dll"}[platform.system()]
    lib_name = "libjumpy_highs" + soext
    pkg_dir = os.path.dirname(__file__)
    repo = os.path.dirname(os.path.dirname(pkg_dir))
    return [
        # Wheel layout: shipped inside the package.
        os.path.join(pkg_dir, "lib", lib_name),
        # Development layout: JuliaC bundle in <repo>/julia/build.
        os.path.join(repo, "julia", "build", "lib", lib_name),
    ]


def find_lib():
    """
    The compiled library _load_lib will use: $JUMPY_LIB if set (authoritative
    even if the file is missing — loading it errors rather than falling back),
    else the first default location that exists, else None.
    """
    import os

    if "JUMPY_LIB" in os.environ:
        return os.environ["JUMPY_LIB"]
    return next((p for p in _default_paths() if os.path.exists(p)), None)


def _load_lib():
    import os

    if _LIB is not None:
        return _LIB
    path = find_lib()
    if path is None:
        raise FileNotFoundError(
            "Could not find the compiled JuMPy backend library. Searched:\n  "
            + "\n  ".join(_default_paths()) + "\nEither:\n"
            "  1. Install the pre-built wheel: pip install jumpy\n"
            "  2. Build it locally: see julia/README.md\n"
            "  3. Use JuliaCall: import jumpy.juliacall as jp\n"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"JUMPY_LIB points to a missing file: {path}")
    return _init_lib(path)


def _init_lib(path):
    global _LIB
    juliacall = sys.modules.get("juliacall")
    if juliacall is not None and juliacall.CONFIG.get("inited"):
        raise RuntimeError(
            "JuliaCall is already initialized. Use one JuMPy backend per process; "
            "restart Python to switch to jumpy.highs."
        )
    # RTLD_GLOBAL so that libjulia symbols are visible process-wide,
    # which the Julia runtime requires.
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    # Check the new ABI before starting Julia; an older build must be rebuilt.
    try:
        lib.jumpy_variables.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.jumpy_variables.restype = ctypes.c_void_p
        lib.jumpy_value.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.jumpy_value.restype = ctypes.c_double
        lib.jumpy_apply.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_longlong]
        lib.jumpy_apply.restype = ctypes.c_void_p
        lib.jumpy_integer_constant.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.jumpy_integer_constant.restype = ctypes.c_void_p
        lib.jumpy_integer_iterator.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_longlong]
        lib.jumpy_integer_iterator.restype = ctypes.c_void_p
        for name in ("jumpy_less_than", "jumpy_greater_than", "jumpy_equal_to"):
            constructor = getattr(lib, name)
            constructor.argtypes = [ctypes.c_double]
            constructor.restype = ctypes.c_uint64
        for name in ("jumpy_zero_one", "jumpy_integer"):
            constructor = getattr(lib, name)
            constructor.argtypes = []
            constructor.restype = ctypes.c_uint64
        lib.jumpy_free_set.argtypes = [ctypes.c_uint64]
        lib.jumpy_free_set.restype = ctypes.c_int
    except AttributeError:
        raise RuntimeError("Rebuild the JuMPy library: this build lacks the native JuMP API") from None

    lib.jl_is_initialized.argtypes = []
    lib.jl_is_initialized.restype = ctypes.c_int
    if lib.jl_is_initialized():
        raise RuntimeError("Julia is already initialized; start a fresh process for jumpy.highs")

    # Initialize the Julia runtime from the image embedded in the library.
    init = lib.jl_init_with_image_handle
    init.argtypes = [ctypes.c_void_p]
    init.restype = None
    init(lib._handle)

    c_longlong = ctypes.c_longlong
    c_int = ctypes.c_int
    c_double = ctypes.c_double
    c_void_p = ctypes.c_void_p
    p_double = ctypes.POINTER(c_double)

    # A model is an opaque pointer to a Julia object, valid until
    # jumpy_free_model. JuMP expressions are opaque pointers built with the
    # constructor entry points; they belong to the model and are freed
    # with it.
    lib.jumpy_new_model.argtypes = []
    lib.jumpy_new_model.restype = c_void_p
    lib.jumpy_free_model.argtypes = [c_void_p]
    lib.jumpy_free_model.restype = c_int
    lib.jumpy_constant.argtypes = [c_void_p, c_double]
    lib.jumpy_constant.restype = c_void_p
    lib.jumpy_iterator.argtypes = [c_void_p, p_double, c_longlong]
    lib.jumpy_iterator.restype = c_void_p
    lib.jumpy_float_array.argtypes = [c_void_p, p_double, c_longlong]
    lib.jumpy_float_array.restype = c_void_p
    lib.jumpy_add_constraint.argtypes = [c_void_p, c_void_p, ctypes.c_uint64]
    lib.jumpy_add_constraint.restype = c_int
    lib.jumpy_set_objective_sense.argtypes = [c_void_p, c_int]
    lib.jumpy_set_objective_sense.restype = c_int
    lib.jumpy_set_objective_function.argtypes = [c_void_p, c_void_p]
    lib.jumpy_set_objective_function.restype = c_int
    lib.jumpy_optimize.argtypes = [c_void_p]
    lib.jumpy_optimize.restype = c_int

    _LIB = lib
    return lib


class JuliacOps:
    """
    The compiled-library implementation of the JuMP operations. Each method is one
    C call into the entry point wrapping the same Julia function the
    juliacall ops call.
    """

    def __init__(self, lib):
        from jumpy._highs_moi import MOI

        self.MOI = MOI
        self._lib = lib
        self._m = lib.jumpy_new_model()
        if not self._m:  # NULL
            raise RuntimeError("Failed to create model")

    def free(self):
        if self._m:
            self._lib.jumpy_free_model(self._m)
            self._m = None

    def _node(self, node):
        if not node:  # NULL
            raise RuntimeError("Failed to build Julia expression")
        return node

    # -- Native Julia objects -------------------------------------------------

    def constant(self, value):
        if isinstance(value, int):
            if not -(2**63) <= value < 2**63:
                raise OverflowError("Integer literals must fit in a Julia Int64")
            return self._node(self._lib.jumpy_integer_constant(self._m, value))
        return self._node(self._lib.jumpy_constant(self._m, value))

    def apply(self, op, args):
        argv = (ctypes.c_void_p * len(args))(*args)
        return self._node(
            self._lib.jumpy_apply(self._m, op.encode(), argv, len(args))
        )

    def iterator(self, values):
        if all(isinstance(value, int) for value in values):
            if any(not -(2**63) <= value < 2**63 for value in values):
                raise OverflowError("Integer iterator values must fit in a Julia Int64")
            data = (ctypes.c_longlong * len(values))(*values)
            return self._node(self._lib.jumpy_integer_iterator(self._m, data, len(values)))
        data = (ctypes.c_double * len(values))(*values)
        return self._node(self._lib.jumpy_iterator(self._m, data, len(values)))

    def float_array(self, values):
        data = (ctypes.c_double * len(values))(*values)
        return self._node(self._lib.jumpy_float_array(self._m, data, len(values)))

    # -- Model building ----------------------------------------------------------

    def add_variables(self, count):
        return self._node(self._lib.jumpy_variables(self._m, count))

    def _set_handle(self, set_):
        from jumpy._highs_moi import NativeSet

        if not isinstance(set_, NativeSet) or set_._lib is not self._lib:
            raise TypeError("Expected an MOI set created by jumpy.highs")
        return set_.handle

    def add_constraint(self, func, set_):
        if self._lib.jumpy_add_constraint(self._m, func, self._set_handle(set_)) != 0:
            raise RuntimeError("Failed to add constraint")

    def set_objective(self, sense, func):
        if self._lib.jumpy_set_objective_sense(self._m, 0 if sense == "min" else 1) != 0:
            raise RuntimeError("Failed to set objective sense")
        if self._lib.jumpy_set_objective_function(self._m, func) != 0:
            raise RuntimeError("Failed to set objective function")

    def optimize(self):
        return self._lib.jumpy_optimize(self._m)

    def value(self, func):
        value = self._lib.jumpy_value(self._m, func)
        if math.isnan(value):
            raise RuntimeError("Failed to query value")
        return value
