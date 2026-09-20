"""
Backend selection and the compiled-library (juliac) MOI ops.

An "ops" object maps each MOI call either to the compiled shared library
(JuliacOps below, via ctypes) or to Julia through juliacall
(jumpy.bridge_juliacall.JuliaCallOps). Models call the ops directly; the
two implementations expose the same methods.
"""

from __future__ import annotations

import ctypes
import atexit
import os
from pathlib import Path


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
_SHUTTING_DOWN = False


def _shutdown():
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True


def _arm_shutdown_guard():
    # Register again after JuliaCall initializes: atexit runs in reverse order,
    # so native finalizers stop calling Julia before JuliaCall tears it down.
    atexit.register(_shutdown)


def _open_lib(path):
    # Retain the GIL when the same runtime also hosts PythonCall.
    lib = ctypes.PyDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    try:
        version = ctypes.c_uint32.in_dll(lib, "jumpy_abi_version").value
        lib._jumpy_trimmed = bool(ctypes.c_uint32.in_dll(lib, "jumpy_image_trimmed").value)
    except ValueError:
        raise RuntimeError("This JuMPy library has no constructor ABI metadata; rebuild with julia/build.jl") from None
    if version != 1:
        raise RuntimeError(f"Unsupported JuMPy constructor ABI version: {version}")
    for name in ("jumpy_runtime_image_path", "jumpy_runtime_library_path", "jl_ver_string"):
        function = getattr(lib, name)
        function.argtypes, function.restype = [], ctypes.c_char_p
    lib.jl_is_initialized.argtypes = []
    lib.jl_is_initialized.restype = ctypes.c_int
    return lib


def _check_active_image(lib):
    active = lib.jumpy_runtime_image_path()
    if active is None or Path(os.fsdecode(active)).resolve() != Path(lib._name).resolve():
        raise RuntimeError(
            "Julia is already initialized with a different system image. "
            "Restart Python and select the same JuMPy shared image for both backends."
        )


def _require_shared_image(lib):
    if lib._jumpy_trimmed:
        raise RuntimeError(
            "A trimmed JuMPy image cannot initialize JuliaCall/PythonCall. "
            "Use the 'shared' build profile for native-object sharing, or use "
            "the trimmed image with backend='juliac' only."
        )


def _check_shared_runtime(lib, jl):
    _require_shared_image(lib)
    from juliacall import CONFIG

    own = ctypes.cast(lib.jl_is_initialized, ctypes.c_void_p).value
    other = ctypes.cast(CONFIG["lib"].jl_is_initialized, ctypes.c_void_p).value
    if own != other:
        raise RuntimeError("JuliaCall and JuMPy loaded different Julia runtimes")
    _check_active_image(lib)
    if not jl.seval("isdefined(Main, :JuMPyHiGHS) && isdefined(JuMPyHiGHS, :JuMPyMOIABI)"):
        raise RuntimeError("JuliaCall's image does not contain the JuMPy constructor ABI")


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
        # Supported development build, then the legacy/CI bundle location.
        os.path.join(repo, "julia", "build-shared", "lib", lib_name),
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

    if _SHUTTING_DOWN:
        raise RuntimeError("The Julia runtime is shutting down")
    if _LIB is not None:
        return _LIB
    path = find_lib()
    if path is None:
        raise FileNotFoundError(
            "Could not find the compiled JuMPy backend library. Searched:\n  "
            + "\n  ".join(_default_paths()) + "\nEither:\n"
            "  1. Install the pre-built wheel: pip install jumpy\n"
            "  2. Build it locally: see julia/README.md\n"
            "  3. Use the juliacall backend: jp.Model(backend='juliacall')\n"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"JUMPY_LIB points to a missing file: {path}")
    return _init_lib(path)


def _init_lib(path):
    global _LIB
    if _SHUTTING_DOWN:
        raise RuntimeError("The Julia runtime is shutting down")
    lib = _open_lib(path)
    if lib.jl_is_initialized():
        _check_active_image(lib)

    # Initialize the Julia runtime from the image embedded in the library.
    init = lib.jl_init_with_image_handle
    init.argtypes = [ctypes.c_void_p]
    init.restype = None
    init(lib._handle)
    _check_active_image(lib)

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

    u64 = ctypes.c_uint64
    p_u64 = ctypes.POINTER(u64)
    for name, arguments, result in (
        ("jumpy_moi_scalar_set", [c_int, c_double], u64),
        ("jumpy_moi_vector_set", [c_int, c_longlong], u64),
        ("jumpy_moi_constant", [c_double], u64),
        ("jumpy_moi_variable", [c_longlong], u64),
        ("jumpy_moi_scalar_nonlinear", [ctypes.c_char_p, p_u64, c_longlong], u64),
        ("jumpy_moi_release", [u64], c_int),
        ("jumpy_moi_kind", [u64], c_int),
        ("jumpy_add_constraint_set", [c_void_p, c_void_p, u64], c_longlong),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result

    _LIB = lib
    _arm_shutdown_guard()
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
        from jumpy.moi import MOIConstructors

        self._lib = lib
        self._constructors = MOIConstructors(lib)
        self._m = lib.jumpy_new_model()
        if not self._m:  # NULL
            raise RuntimeError("Failed to create model")

    def free(self):
        if self._m:
            if not _SHUTTING_DOWN:
                self._lib.jumpy_free_model(self._m)
            self._m = None

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
        with self._constructors.scalar_set(sense, rhs) as set_:
            ci = self._lib.jumpy_add_constraint_set(self._m, func, set_.handle)
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
