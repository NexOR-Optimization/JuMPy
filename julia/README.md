# JuMPyHiGHS

The compiled Julia backend for JuMPy exposes native
[JuMP](https://github.com/jump-dev/JuMP.jl) and
[GenOpt](https://github.com/blegat/GenOpt.jl) modeling operations through a small
C ABI. Python holds opaque references and forwards operations; Julia constructs
and classifies expressions.

The compiled backend creates a concrete JuMP direct model over raw HiGHS: no
caching optimizer or bridge optimizer. JuMP constructs and normalizes native
expressions and constraints; the thin `JuMPyDirect` adapter sends them to
`JuMP.backend(model)` through MOI. GenOpt's `add_constraints` expands generated
constraints directly. The same expansion helper is used by GenOpt's bridge;
JuMPy does not implement another expansion loop. HiGHS supports linear
constraints and convex quadratic objectives. Unsupported functions and sets
produce errors.

JuliaCall instead creates a regular JuMP model with caching and bridges,
including GenOpt's `FunctionGeneratorBridge`. Both frontends use the same
native JuMP/GenOpt expressions; their solver plumbing is intentionally different.

In Python, select this backend with `import jumpy.highs as jp`, then construct
`jp.Model()` without a backend argument. `import jumpy.juliacall as jp` selects
the alternative JuliaCall interface. The two interfaces share modeling
source, not live Julia objects; use one runtime per process.

Both support explicit scalar sets alongside comparison syntax:

```python
import jumpy.highs as jp

model = jp.Model()
x, y = model.variables(2, lower=0)
model.constraint(x + y, jp.MOI.LessThan(1.0))
```

The compiled facade exposes the five native scalar sets `LessThan`,
`GreaterThan`, `EqualTo`, `ZeroOne`, and `Integer`. Python stores only ownership
handles, not duplicate set definitions. JuliaCall exposes the actual MOI module.

## C ABI

See `src/JuMPyHiGHS.jl` for the full conventions.

Rebuild existing compiled libraries for this ABI. The old column-index and
bulk-solution entry points are removed; the Python loader rejects binaries
without the native array and value-query entry points before starting Julia.

A model is an opaque `void*` pointing at the Julia-side model object; native
expressions are opaque `void*` nodes built with the operation entry points.
Both are rooted on the Julia side (the GC cannot see references held by C):
nodes belong to the model that built them, and everything is freed by
`jumpy_free_model`, after which no pointer from that model may be used.

| Function | Julia operation |
|---|---|
| `jumpy_new_model() -> void*` | Create a concrete JuMP direct model over raw HiGHS (NULL on error) |
| `jumpy_free_model(m) -> int32` | release the model and its nodes |
| `jumpy_variables(m, count) -> void*` | Add JuMP variables and return their native array reference |
| `jumpy_constant(m, value) -> void*` | a `Float64` node |
| `jumpy_integer_constant(m, value) -> void*` | an `Int64` node, preserving integer exponents and indices |
| `jumpy_apply(m, op, args**, nargs) -> void*` | Apply the native Julia operator to JuMP/GenOpt arguments |
| `jumpy_iterator(m, values*, len) -> void*` | `GenOpt.iterator` over `Float64` values |
| `jumpy_integer_iterator(m, values*, len) -> void*` | `GenOpt.iterator` over `Int64` values, including integer index ranges |
| `jumpy_float_array(m, values*, len) -> void*` | a data vector, 1-based-indexable in templates |
| `jumpy_less_than(rhs) -> uint64` | Construct and retain `MOI.LessThan(rhs)` |
| `jumpy_greater_than(rhs) -> uint64` | Construct and retain `MOI.GreaterThan(rhs)` |
| `jumpy_equal_to(rhs) -> uint64` | Construct and retain `MOI.EqualTo(rhs)` |
| `jumpy_zero_one() -> uint64` | Construct and retain `MOI.ZeroOne()` |
| `jumpy_integer() -> uint64` | Construct and retain `MOI.Integer()` |
| `jumpy_free_set(id) -> int32` | Release a native set handle |
| `jumpy_add_constraint(m, f, id) -> int32` | Build and add a JuMP constraint using the native set, including GenOpt templates; 0 on success, -1 on error |
| `jumpy_set_objective_sense(m, sense) -> int32` | Set the JuMP objective sense; 0 = min, 1 = max |
| `jumpy_set_objective_function(m, f) -> int32` | Set the native JuMP objective expression |
| `jumpy_optimize(m) -> int32` | Optimize the raw MOI backend; returns the termination status (`OPTIMAL == 1`) |
| `jumpy_set_time_limit(m, seconds) -> int32` | Set a finite nonnegative time limit; NaN clears it; 0 on success, -1 on error |
| `jumpy_value(m, expr) -> float64` | Evaluate the native expression using MOI variable values; NaN on error |

Arrays and their indexed variables are ordinary native expression references.
Python translates local zero-based indices to Julia's one-based `getindex`
through `jumpy_apply`; no global column IDs or variable reconstruction are
needed. Value queries use the expression reference directly, with no Python
solution-vector cache. Constraint insertion returns only a success status,
not a retained constraint ID.

Native sets use checked `uint64` identity handles, not set-kind codes; constructors
return `0` on error. Comparisons, variable bounds, and constraint groups all use
these native sets. A set can be reused across models in the same image and remains valid
after a model closes, until the set is released. The Python wrapper releases
sets automatically and also supports `close()` and context managers. Handles
cannot be passed to JuliaCall or another image.

Python exposes `model.set_time_limit_sec(seconds)` and
`model.unset_time_limit_sec()` for setting and clearing time limits. Both use
the existing `jumpy_set_time_limit` transport in the compiled backend; NaN is
an internal clearing sentinel, not a public argument.

The shared expression operations and JuliaCall model implementation live in
[`src/JuMPyModel.jl`](src/JuMPyModel.jl). It is a
solver-independent Julia module shared with the JuliaCall backend, and is
included in the Python wheel so JuliaCall can load it without a source checkout.
The JuliaC backend includes the same file at build time, plus its bridge-free
[`src/JuMPyDirect.jl`](src/JuMPyDirect.jl) adapter. Model ownership,
opaque-pointer rooting, and C entry points remain in `JuMPyHiGHS`.
There is no additional shared object or second Julia runtime to initialize.
Keeping the canonical source inside the Julia project also lets JuliaC copy
that project into an isolated build directory. Wheel builds package the same
file as `jumpy/julia/JuMPyModel.jl`; editable installs load the canonical source.

The consumer must initialize the Julia runtime once after loading the
library, by calling `jl_init_with_image_handle(dlopen_handle)` (see
`_load_lib` in `src/jumpy/backend.py`).

## Archived experiments

Historical shared-library and trimming probes are preserved as-is on the
`bl/experiments` branch under `julia/experiments/`. They are not part of the
maintained backend or its CI.

## Building

The build uses Julia 1.13.2, a C compiler, and the
[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) frontend:

```bash
julia --project=@juliac -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaC", version="0.3.12"))'
```

This integration requires the patched development checkouts of JuMP and GenOpt.
JuMP provides concrete model/reference types through `concrete_direct_model`;
GenOpt provides direct generator expansion and concrete-reference-compatible
symbolic indexing. The compiled projects and manifests select JuMP's
`bl/juliac` and GenOpt's `bl/juliac-safe` branches, with registered HiGHS
1.26.1 or newer. The local setup below can instead use development checkouts.
JuliaCall needs the GenOpt changes, but can use released JuMP because it does
not construct a concrete direct model.

From this directory, prepare a separate Julia environment so local development
paths do not change the repository manifest:

```bash
export JUMPY_GENOPT_SOURCE=/path/to/patched/GenOpt
export JUMPY_JUMP_SOURCE=/path/to/patched/JuMP
export JUMPY_BACKEND_SOURCE="$PWD"
export JUMPY_JULIA_PROJECT="$(mktemp -d)"
cp Manifest.toml "$JUMPY_JULIA_PROJECT/Manifest.toml"
JULIA_PKG_OFFLINE=true julia --project="$JUMPY_JULIA_PROJECT" -e '
    using Pkg
    Pkg.develop([
        PackageSpec(path=ENV["JUMPY_GENOPT_SOURCE"]),
        PackageSpec(path=ENV["JUMPY_JUMP_SOURCE"]),
        PackageSpec(path=ENV["JUMPY_BACKEND_SOURCE"]),
    ])
    Pkg.instantiate()
'
```

Offline mode requires the other dependencies and solver artifacts to be cached;
omit `JULIA_PKG_OFFLINE=true` if they need downloading. Then build with that
environment. CI builds without trimming; the `--trim=unsafe-warn` flag below
opts into experimental trimming:

```bash
JULIA_CPU_TARGET=generic julia --project=@juliac -m JuliaC \
    --output-lib build/libjumpy_highs --project "$JUMPY_JULIA_PROJECT" \
    --compile-ccallable \
    --trim=unsafe-warn \
    --jl-option handle-signals=no \
    --bundle build \
    juliac_entry.jl
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
- Trimming uses the earlier `--trim=unsafe-warn` approach with explicit roots in
  `trim_roots.jl` and supplementary dynamic-dispatch roots in `trim_dispatch.jl`.
  Compiler warnings remain: this is not a `--trim=safe` guarantee. Validate the
  compiled artifact with the full Python tests and executable tutorials.
- The dispatch roots depend on the Julia/dependency versions. Regenerate them
  from the ordinary-Julia workload when those versions or the ABI change;
  instructions are in [`generate_trim_dispatch.jl`](generate_trim_dispatch.jl).
  The checked roots use HiGHS 1.26.1 and MOI 1.54.0, as in the repository
  manifest, plus the development changes above. The image-build entry
  script does not execute a solver workload because JLL initialization has not
  run at that point.

## Testing

In-process tests of the entry points (fast, no compilation):

```bash
julia --project="$JUMPY_JULIA_PROJECT" test/runtests.jl
```

End-to-end through the compiled library and Python ctypes:

```bash
cd ..
JUMPY_BACKEND=juliac uv run --group tests pytest tests/test_solve.py
```

The Python loader searches `$JUMPY_LIB`, the installed package's `lib/`
directory, then `julia/build/lib/` (this development layout).

For JuliaCall, use a PythonCall environment with the same patched GenOpt
checkout developed into it. For an existing environment:

```bash
export JUMPY_PYTHONCALL_PROJECT=/path/to/pythoncall/environment
JULIA_PKG_OFFLINE=true julia --project="$JUMPY_PYTHONCALL_PROJECT" -e '
    using Pkg
    Pkg.develop(path=ENV["JUMPY_GENOPT_SOURCE"])
'
export PYTHON_JULIACALL_PROJECT="$JUMPY_PYTHONCALL_PROJECT"
export PYTHON_JULIACALL_EXE="$(julia -e 'print(joinpath(Sys.BINDIR, Base.julia_exename()))')"
```
