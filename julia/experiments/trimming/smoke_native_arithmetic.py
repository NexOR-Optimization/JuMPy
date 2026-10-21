"""Run directly with the arithmetic .so path; requires a fresh Python process."""

import ctypes
import itertools
import sys


library = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
library.jl_init_with_image_handle.argtypes = [ctypes.c_void_p]
library.jl_init_with_image_handle.restype = None
library.jl_init_with_image_handle(library._handle)
library.probe_node.argtypes = [ctypes.c_int]
library.probe_node.restype = ctypes.c_void_p
library.probe_apply.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
library.probe_apply.restype = ctypes.c_void_p
library.probe_evaluate.argtypes = [ctypes.c_void_p]
library.probe_evaluate.restype = ctypes.c_double

x, two, one = (library.probe_node(index) for index in (1, 2, 3))
assert x and two and one
affine = library.probe_apply(b"+", x, one)
assert affine
operands = [(x, 3.0), (two, 2.0), (one, 1.0), (affine, 4.0)]
count = 0
for op, (left, a), (right, b) in itertools.product((b"+", b"*"), operands, operands):
    result = library.probe_apply(op, left, right)
    assert result, (op, a, b)
    expected = a + b if op == b"+" else a * b
    assert library.probe_evaluate(result) == expected, (op, a, b)
    count += 1

# Quadratic outputs remain real JuMP expressions; this bounded arithmetic ABI
# intentionally accepts only scalar/variable/affine inputs and rejects them.
quadratic = library.probe_apply(b"*", affine, affine)
assert library.probe_evaluate(quadratic) == 16.0
assert not library.probe_apply(b"+", quadratic, one)
assert not library.probe_apply(b"unknown", x, one)
# Rejected calls must not damage subsequent native expression construction.
assert library.probe_evaluate(library.probe_apply(b"+", x, two)) == 5.0
print(f"PASS: {count} native +/* cases, quadratic evaluation, two clean rejections, post-error reuse")
