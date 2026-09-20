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

The default build contains HiGHS, GenOpt, and the constructors from
[`src/JuMPyMOI.jl`](src/JuMPyMOI.jl) in **one Julia image**. JuliaCall can attach
to that same image/runtime and use the actual MOI values. Loading independent
Julia images does not make their objects interchangeable.

The additional ABI in [`src/JuMPyMOIABI.jl`](src/JuMPyMOIABI.jl) owns model-independent
sets and function nodes in a Julia registry:

| Entry point | Result |
|---|---|
| `jumpy_moi_scalar_set(sense, rhs)` | Native `LessThan`, `GreaterThan`, `EqualTo`, `ZeroOne`, or `Integer` |
| `jumpy_moi_vector_set(sense, dimension)` | Native `Nonpositives`, `Nonnegatives`, or `Zeros` |
| `jumpy_moi_constant(value)`, `jumpy_moi_variable(index)` | Function leaves; variable indices are MOI's **one-based** indices |
| `jumpy_moi_scalar_nonlinear(head, args*, nargs)` | Native nonlinear function retaining its argument objects |
| `jumpy_moi_release(handle)` | Release the registry's reference |
| `jumpy_moi_kind(handle)` | Inspect the supported kind; `-1` for an invalid handle |
| `jumpy_add_constraint_set(m, f, handle)` | Consume a native scalar set with the existing model/function-pointer ABI |

Constructors return checked `uint64` handles (`0` on failure), not Julia object
addresses. Handles are image-local, never reused, and invalid after release.
Releasing a model-independent handle does not invalidate objects already owned
by an expression or borrowed by JuliaCall. No Python class duplicates a set's
Julia definition.

The Python loader validates native ABI/profile metadata before initialization,
rejects a different active image, and initializes the runtime only once. Build
through `build.jl` so this metadata is included. The constructor source is also
packaged as `jumpy/julia/JuMPyMOI.jl` for source-only JuliaCall use when no compiled
library is available.

## Building

Requires Julia, a C compiler, and the
[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) frontend:

```bash
julia --project=@juliac -e 'using Pkg; Pkg.add("JuliaC")'
```

Then, from this directory:

```bash
julia --project=@juliac build.jl shared
```

`shared` is the default profile. An optional second argument selects the output
directory, for example `build.jl shared build`. The default output is
`build-shared/lib/libjumpy_highs.so` (`.dylib`/`.dll` on other platforms).
The wrapper disables Julia's signal handlers and bundles the runtime libraries
and HiGHS artifacts. HiGHS's native solver library is a dependency of this one
Julia image, not a second Julia image.

### Using JuliaCall with the shared image

Install `jumpy[juliacall]` and select the same Julia version used to build the
image. `PYTHON_JULIAPKG_EXE` can select the matching Julia executable;
`PYTHON_JULIAPKG_PROJECT` can select a compatible environment containing
PythonCall. Start a fresh Python process and let JuMPy configure JuliaCall:
importing ordinary JuliaCall first initializes a different image, which JuMPy
will reject. Both JuliaCall-first **through JuMPy** and compiled-first use are
supported. The compiled-only backend never needs to import JuliaCall.

For low-level native construction:

```python
from jumpy.moi import MOIConstructors

native = MOIConstructors()
with native.scalar_set("<=", 1.0) as owned:
    set_ = owned.to_julia()  # Actual MOI.LessThan{Float64} in the same runtime
    assert set_.upper == 1.0
# The registry handle is released; JuliaCall still owns its borrowed value.
assert set_.upper == 1.0
```

`to_julia()` requires the optional JuliaCall dependency. Construction and handle
release alone do not. `from_julia(value)` roots a supported object from the same
runtime. This low-level API does **not** add `Model.constraint(func, set)` yet.

### Experimental trimming

To try the compiled-only profile:

```bash
julia --project=@juliac build.jl trimmed
```

Select it explicitly with `JUMPY_LIB`; `build-trimmed` is not auto-discovered.
Do not load both profiles in one process. A trimmed image cannot initialize
JuliaCall/PythonCall, and the loader rejects that combination before import.

This profile uses `--trim=unsafe-warn` and finite entry-point declarations in
`juliac_entry.jl`, `trim_dispatch.jl`, and `trim_runtime.jl`. Tests and tutorials
exercise supported combinations, but verifier warnings remain: successful
examples are not a guarantee for arbitrary models or dependency versions.
Missing dynamic-dispatch specializations can still fail at runtime. The shared,
untrimmed image is the primary build.

## Testing

In-process tests of the entry points (fast, no compilation):

```bash
julia --project=. test/runtests.jl
```

End-to-end through the compiled library and Python ctypes:

```bash
cd ..
JUMPY_BACKEND=juliac uv run --group tests pytest
```

`JUMPY_LIB` overrides discovery. Otherwise, the Python loader searches the
installed package's `lib/`, then `julia/build-shared/lib/`, then the legacy
`julia/build/lib/` development location.
