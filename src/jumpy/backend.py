"""
Backend selection and the compiled-library (juliac) MOI ops.

An "ops" object maps each MOI call either to the compiled shared library
(JuliacOps below, via ctypes) or to Julia through juliacall
(jumpy.bridge_juliacall.JuliaCallOps). Models call the ops directly; the
two implementations expose the same methods.
"""

from __future__ import annotations

import ctypes


def get_ops(backend):
    if backend == "juliac":
        return JuliacOps(_load_lib())
    if backend == "juliacall":
        from jumpy.bridge_juliacall import JuliaCallOps

        return JuliaCallOps()
    if isinstance(backend, str):
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: juliac, juliacall"
        )
    # An ops object used directly (tests, custom backends).
    return backend


# The Julia runtime can only be initialized once per process.
_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    import os
    import platform

    soext = {"Linux": ".so", "Darwin": ".dylib", "Windows": ".dll"}[platform.system()]
    lib_name = "libjumpy_highs" + soext

    # JUMPY_LIB is authoritative: no silent fallback to another library.
    if "JUMPY_LIB" in os.environ:
        path = os.environ["JUMPY_LIB"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"JUMPY_LIB points to a missing file: {path}")
        return _init_lib(path)

    candidates = []
    pkg_dir = os.path.dirname(__file__)
    # Wheel layout: shipped inside the package.
    candidates.append(os.path.join(pkg_dir, "lib", lib_name))
    # Development layout: JuliaC bundle in <repo>/julia/build.
    repo = os.path.dirname(os.path.dirname(pkg_dir))
    candidates.append(os.path.join(repo, "julia", "build", "lib", lib_name))

    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not find the compiled JuMPy backend library ({lib_name}). "
            "Searched:\n  " + "\n  ".join(candidates) + "\nEither:\n"
            "  1. Install the pre-built wheel: pip install jumpy\n"
            "  2. Build it locally: see julia/README.md\n"
            "  3. Use the juliacall backend: jp.Model(backend='juliacall')\n"
        )
    return _init_lib(path)


def _init_lib(path):
    global _LIB
    # RTLD_GLOBAL so that libjulia symbols are visible process-wide,
    # which the Julia runtime requires.
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

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
    p_void = ctypes.POINTER(c_void_p)

    # A model is an opaque pointer to a Julia object, valid until
    # jumpy_free_model. MOI functions are opaque pointers built with the
    # constructor entry points; they belong to the model and are freed
    # with it.
    lib.jumpy_new_model.argtypes = []
    lib.jumpy_new_model.restype = c_void_p
    lib.jumpy_free_model.argtypes = [c_void_p]
    lib.jumpy_free_model.restype = c_int
    lib.jumpy_add_variables.argtypes = [c_void_p, c_longlong]
    lib.jumpy_add_variables.restype = c_longlong
    lib.jumpy_constant.argtypes = [c_void_p, c_double]
    lib.jumpy_constant.restype = c_void_p
    lib.jumpy_variable.argtypes = [c_void_p, c_longlong]
    lib.jumpy_variable.restype = c_void_p
    lib.jumpy_scalar_nonlinear.argtypes = [c_void_p, ctypes.c_char_p, p_void, c_longlong]
    lib.jumpy_scalar_nonlinear.restype = c_void_p
    lib.jumpy_iterator.argtypes = [c_void_p, p_double, c_longlong]
    lib.jumpy_iterator.restype = c_void_p
    lib.jumpy_contiguous_variables.argtypes = [c_void_p, c_longlong, c_longlong]
    lib.jumpy_contiguous_variables.restype = c_void_p
    lib.jumpy_float_array.argtypes = [c_void_p, p_double, c_longlong]
    lib.jumpy_float_array.restype = c_void_p
    lib.jumpy_add_constraint.argtypes = [c_void_p, c_void_p, c_int, c_double]
    lib.jumpy_add_constraint.restype = c_longlong
    lib.jumpy_add_group_constraint.argtypes = [c_void_p, c_void_p, c_int]
    lib.jumpy_add_group_constraint.restype = c_longlong
    lib.jumpy_set_objective_sense.argtypes = [c_void_p, c_int]
    lib.jumpy_set_objective_sense.restype = c_int
    lib.jumpy_set_objective_function.argtypes = [c_void_p, c_void_p]
    lib.jumpy_set_objective_function.restype = c_int
    lib.jumpy_optimize.argtypes = [c_void_p]
    lib.jumpy_optimize.restype = c_int
    lib.jumpy_primal_status.argtypes = [c_void_p]
    lib.jumpy_primal_status.restype = c_int
    lib.jumpy_get_values.argtypes = [c_void_p, p_double, c_longlong]
    lib.jumpy_get_values.restype = c_longlong
    lib.jumpy_objective_value.argtypes = [c_void_p]
    lib.jumpy_objective_value.restype = c_double

    _LIB = lib
    return lib


# Set codes of jumpy_add_constraint: {0: LessThan, 1: GreaterThan,
# 2: EqualTo}(rhs), {3: ZeroOne, 4: Integer} (rhs ignored).
_SENSE_CODES = {"<=": 0, ">=": 1, "==": 2, "binary": 3, "integer": 4}


class JuliacOps:
    """
    The compiled-library implementation of the MOI ops. Each method is one
    C call into the entry point wrapping the same MOI function the
    juliacall ops call.
    """

    def __init__(self, lib):
        self._lib = lib
        self._m = lib.jumpy_new_model()
        if not self._m:  # NULL
            raise RuntimeError("Failed to create model")

    def free(self):
        self._lib.jumpy_free_model(self._m)

    def _node(self, node):
        if not node:  # NULL
            raise RuntimeError("Failed to build MOI function")
        return node

    # -- MOI functions ---------------------------------------------------------

    def constant(self, value):
        return self._node(self._lib.jumpy_constant(self._m, value))

    def variable(self, index):
        return self._node(self._lib.jumpy_variable(self._m, index))

    def scalar_nonlinear(self, head, args):
        argv = (ctypes.c_void_p * len(args))(*args)
        return self._node(
            self._lib.jumpy_scalar_nonlinear(self._m, head.encode(), argv, len(args))
        )

    def iterator(self, values):
        data = (ctypes.c_double * len(values))(*values)
        return self._node(self._lib.jumpy_iterator(self._m, data, len(values)))

    def contiguous_variables(self, start, count):
        return self._node(self._lib.jumpy_contiguous_variables(self._m, start, count))

    def float_array(self, values):
        data = (ctypes.c_double * len(values))(*values)
        return self._node(self._lib.jumpy_float_array(self._m, data, len(values)))

    # -- Model building ----------------------------------------------------------

    def add_variables(self, count):
        start = self._lib.jumpy_add_variables(self._m, count)
        if start < 0:
            raise RuntimeError("Failed to add variables")
        return start

    def add_constraint(self, func, sense, rhs):
        ci = self._lib.jumpy_add_constraint(self._m, func, _SENSE_CODES[sense], rhs)
        if ci < 0:
            raise RuntimeError("Failed to add constraint")

    def add_constraint_group(self, func, sense, linear):
        n = self._lib.jumpy_add_group_constraint(self._m, func, _SENSE_CODES[sense])
        if n < 0:
            raise RuntimeError("Failed to add constraint group")

    def set_objective(self, sense, func):
        if self._lib.jumpy_set_objective_sense(self._m, 0 if sense == "min" else 1) != 0:
            raise RuntimeError("Failed to set objective sense")
        if self._lib.jumpy_set_objective_function(self._m, func) != 0:
            raise RuntimeError("Failed to set objective function")

    def optimize(self):
        return self._lib.jumpy_optimize(self._m)

    def get_values(self, count):
        out = (ctypes.c_double * count)()
        written = self._lib.jumpy_get_values(self._m, out, count)
        if written != count:
            raise RuntimeError(f"Expected {count} solution values, got {written}")
        return list(out)
