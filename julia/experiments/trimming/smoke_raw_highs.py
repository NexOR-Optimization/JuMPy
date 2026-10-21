"""Run directly with the raw HiGHS .so path; requires a fresh Python process."""

import ctypes
import math
import sys

lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
lib.jl_is_initialized.argtypes = []
lib.jl_is_initialized.restype = ctypes.c_int
assert lib.jl_is_initialized() == 0
lib.jl_init_with_image_handle.argtypes = [ctypes.c_void_p]
lib.jl_init_with_image_handle.restype = None
lib.jl_init_with_image_handle(lib._handle)
assert lib.jl_is_initialized() == 1
lib.raw_highs_lp.argtypes = [ctypes.c_double]
lib.raw_highs_lp.restype = ctypes.c_double
for rhs in (-3.25, 0.0, 2.5, 100.0):
    actual = lib.raw_highs_lp(rhs)
    assert math.isclose(actual, rhs, abs_tol=1e-9), (rhs, actual)
    print(f"rhs={rhs}: optimum={actual}", flush=True)
print("Fresh ctypes process: four trimmed LP solves passed", flush=True)
