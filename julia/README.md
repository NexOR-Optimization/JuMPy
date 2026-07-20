# JuMPyHiGHS

The compiled Julia backend for JuMPy: C entry points that mirror the
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) API,
one MOI call per entry point. Python builds models by calling these
eagerly (`src/jumpy/expressions.py` + `model.py`), identically for the
juliacall and juliac backends; `jumpy_scalar_nonlinear` is simply the
compiled counterpart of `jl.MOI.ScalarNonlinearFunction(...)`. Nothing is
HiGHS-specific except the `Optimizer` constant in `src/JuMPyHiGHS.jl`: any
MOI optimizer can be compiled behind the same entry points.

The optimizer is used raw — **no** `MOI.Bridges`, **no**
`CachingOptimizer`. Whatever it does not support (for HiGHS: nonlinear
functions) is reported as an error. GenOpt is compiled in: templates
reference iterators by identity (`GenOpt.IteratorRef`) and
`jumpy_add_group_constraint` expands them here with the same loop as
`GenOpt.FunctionGeneratorBridge`, one scalar constraint per combination of
iterator values.

## C ABI

See `src/JuMPyHiGHS.jl` for the full conventions.

A model is an opaque `void*` pointing at the Julia-side model object; MOI
functions are opaque `void*` nodes built with the constructor entry points.
Both are rooted on the Julia side (the GC cannot see references held by C):
nodes belong to the model that built them, and everything is freed by
`jumpy_free_model`, after which no pointer from that model may be used.

| Function | MOI equivalent |
|---|---|
| `jumpy_new_model() -> void*` | `Optimizer()` (NULL on error) |
| `jumpy_free_model(m) -> int32` | release the model and its nodes |
| `jumpy_add_variables(m, count) -> int64` | `MOI.add_variables`; returns the 0-based start index |
| `jumpy_constant(m, value) -> void*` | a `Float64` node |
| `jumpy_variable(m, index) -> void*` | `MOI.VariableIndex` of the 0-based column `index` |
| `jumpy_scalar_nonlinear(m, head, args**, nargs) -> void*` | `MOI.ScalarNonlinearFunction(Symbol(head), Any[args...])` |
| `jumpy_iterator(m, values*, len) -> void*` | `GenOpt.IteratorRef(GenOpt.Iterator(values))`, usable in template expressions |
| `jumpy_contiguous_variables(m, offset, count) -> void*` | `GenOpt.ContiguousArrayOfVariables`, 1-based-indexable block of variables |
| `jumpy_float_array(m, values*, len) -> void*` | a data vector, 1-based-indexable in templates |
| `jumpy_add_constraint(m, f, sense, rhs) -> int64` | `MOI.add_constraint(f, set)` with set `{0: LessThan, 1: GreaterThan, 2: EqualTo}(rhs)` or `{3: ZeroOne, 4: Integer}`; function constants are normalized into the set; variable bounds are just variable nodes |
| `jumpy_add_group_constraint(m, f, sense) -> int64` | expand the template over its iterators (GenOpt), one scalar constraint each; returns the count |
| `jumpy_set_objective_sense(m, sense) -> int32` | `MOI.set(MOI.ObjectiveSense())`; 0 = min, 1 = max |
| `jumpy_set_objective_function(m, f) -> int32` | `MOI.set(MOI.ObjectiveFunction{F}(), f)` |
| `jumpy_optimize(m) -> int32` | `MOI.optimize!`; returns `Int(MOI.TerminationStatusCode)` (`OPTIMAL == 1`) |
| `jumpy_primal_status(m) -> int32` | `Int(MOI.ResultStatusCode)` (`FEASIBLE_POINT == 1`) |
| `jumpy_get_values(m, out*, len) -> int64` | `MOI.VariablePrimal`; copies into `out` |
| `jumpy_objective_value(m) -> float64` | `MOI.ObjectiveValue` |

Affine expressions built as `ScalarNonlinearFunction` trees are narrowed to
`ScalarAffineFunction` with `MOI.Nonlinear.SymbolicAD.simplify` before being
passed to the optimizer, so HiGHS accepts them.

The consumer must initialize the Julia runtime once after loading the
library, by calling `jl_init_with_image_handle(dlopen_handle)` (see
`_load_lib` in `src/jumpy/backend.py`).

## Building

Requires Julia 1.12+, a C compiler, and the
[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) frontend:

```bash
julia --project=@juliac -e 'using Pkg; Pkg.add("JuliaC")'
```

Then, from this directory:

```bash
julia --project=@juliac -m JuliaC \
    --output-lib build/libjumpy_highs \
    --compile-ccallable \
    --jl-option handle-signals=no \
    --bundle build \
    .
```

Notes:

- `--jl-option handle-signals=no` is required because the library is loaded
  into a Python process; Julia's signal handlers would conflict with Python's.
- `--bundle` makes the output relocatable: `build/lib/` contains
  `libjumpy_highs.so` next to the Julia runtime libraries
  (`build/lib/julia/`), and `build/share/julia/artifacts/` contains the
  HiGHS_jll artifact with `libhighs.so`. This is the two-shared-library
  layout: `libjumpy_highs.so` (our entry points + Julia runtime image) loads
  `libhighs.so` (the solver distributed by HiGHS_jll) dynamically.
- No `--trim` for now: the MOI wrapper relies on dynamic dispatch that
  `--trim=safe` cannot verify yet. The untrimmed library is large but
  correct; trimming is an optimization to revisit.

## Testing

In-process tests of the entry points (fast, no compilation):

```bash
julia --project=. test/runtests.jl
```

End-to-end through the compiled library and Python ctypes:

```bash
cd .. && JUMPY_BACKEND=juliac python3 tests/test_solve.py
```

The Python loader searches `$JUMPY_LIB`, the installed package's `lib/`
directory, then `julia/build/lib/` (this development layout).
