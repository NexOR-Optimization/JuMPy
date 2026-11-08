# JuMPy (Julia)

The Julia package `JuMPy` turns a MathOptInterface solver into a compiled
backend of the `jumpy` Python package. A backend exposes native
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

JuliaCall's `jp.Model()` creates a regular cached JuMP model without a solver.
`jp.Model(optimizer_constructor)` forwards a zero-argument optimizer factory to
`JuMP.Model`; this can be a Julia constructor such as `jl.HiGHS.Optimizer` or a
Python lambda. Cached models register GenOpt's `FunctionGeneratorBridge` and
`PolynomialFunctionBridge`. `jp.direct_model(optimizer)` forwards an already
constructed optimizer to `JuMP.direct_model`, without adding a cache or bridges.
Solver-output defaults also follow JuMP. Both frontends use the same native
JuMP/GenOpt expressions; their solver plumbing is intentionally different.
JuliaCall users install and import their selected solver explicitly.

In Python, select a backend with `import jumpy.highs as jp`, then construct
`jp.Model()` without a backend argument. `import jumpy.juliacall as jp` selects
the alternative JuliaCall interface. The two interfaces share modeling
source, not live Julia objects; use one runtime per process.

Both support explicit sets alongside comparison syntax:

```python
import jumpy.highs as jp

model = jp.Model()
x, y = model.variables(2, lower=0)
model.constraint(x + y, jp.MOI.LessThan(1.0))
```

A compiled backend exposes the sets that its optimizer supports, listed in
its registry (`jumpy_registry`): `jp.MOI` holds those of MathOptInterface,
such as `LessThan`, `Interval` or `ZeroOne` for HiGHS, and the sets of other
modules are in namespaces of the same name, such as `jp.MathOptVRP.Partition`
for Vroom. These namespaces also hold the `JuMP.NonlinearOperator`s of these
modules, such as `jp.MathOptVRP.op_sum_distances`. Python stores only ownership
handles, not duplicate set definitions. JuliaCall exposes the actual MOI module.

## Backends

A backend is a recipe: a Julia package `JuMPy<Name>` whose module calls
[`JuMPy.@entrypoints`](src/JuMPy.jl) with the optimizer of its solver, for
example [`backends/highs`](backends/highs):

```julia
module JuMPyHiGHS
import HiGHS
import JuMPy
JuMPy.@entrypoints HiGHS.Optimizer
end
```

The recipe directory also contains

- `workload.jl`, which exercises the C ABI through `JuMPy.Client`, as Python
  does. It is traced in ordinary Julia to find the methods reached through
  dynamic dispatch, which the image must retain. `JuMPy.Client.workload`
  exercises the expressions and scalar sets of any backend; a solver of other
  sets, such as a routing solver, builds models of its own;
- optionally, `roots.jl`, which registers other compilation roots with
  `Base.Experimental.entrypoint` in the image-build session.

The sets of the C ABI are those that the optimizer declares in its
`MOI.supports_constraint`, `MOI.supports_add_constrained_variable` and
`MOI.supports_add_constrained_variables` methods, whose fields are integers,
floats, booleans, or vectors and matrices of these. A set type `S` is
constructed from its fields as `S(fields...)`. A recipe can also list them:
`JuMPy.@entrypoints(NexOR.Optimizer, (MOI.LessThan{Float64}, ...))`. The
operators of the C ABI are the `JuMP.NonlinearOperator`s of the modules of
these sets.

[`build_backend.py`](build_backend.py) compiles a recipe and packages it for
Python. The `jumpy-opt` wheel ships the HiGHS backend, built from
[`backends/highs`](backends/highs). Another backend is distributed as its own
wheel `jumpy-opt-<name>`, which installs the module `jumpy/<name>.py` and the
bundle `jumpy/_backends/<name>/`, without Python code to write:

```bash
python julia/build_backend.py path/to/recipe --wheel dist  # dist/jumpy_opt_<name>-*.whl
```

For example, the recipe of the [Vroom](https://github.com/NexOR-Optimization/Vroom.jl)
routing solver is in its `jumpy/` directory, and is used from Python as

```python
import jumpy.vroom as jp

model = jp.Model()
nodes = model.variables((4, 2), jp.MathOptVRP.Partition(4, 2))
model.objective = jp.minimize(sum(
    jp.MathOptVRP.op_sum_distances(distances, jp.vcat(5, nodes[:, j], 5))
    for j in range(2)
))
model.optimize()
```

## C ABI

See `src/entrypoints.jl` for the full conventions.

The version of the ABI is `JuMPy.ABI_VERSION`, returned by
`jumpy_abi_version`. The Python package only loads libraries of the version it
expects, `jumpy.backend.ABI_VERSION`: rebuild existing compiled libraries when
it changes.

A model is an opaque `void*` pointing at the Julia-side model object; native
expressions are opaque `void*` nodes built with the operation entry points.
Both are rooted on the Julia side (the GC cannot see references held by C):
nodes belong to the model that built them, and everything is freed by
`jumpy_free_model`, after which no pointer from that model may be used.

| Function | Julia operation |
|---|---|
| `jumpy_abi_version() -> int64` | The version of this ABI, `JuMPy.ABI_VERSION` |
| `jumpy_registry() -> char*` | The sets and operators of the backend, as tab-separated lines (see below) |
| `jumpy_new_model() -> void*` | Create a concrete JuMP direct model over the raw optimizer (NULL on error) |
| `jumpy_free_model(m) -> int32` | release the model and its nodes |
| `jumpy_variables(m, count) -> void*` | Add JuMP variables and return their native array reference |
| `jumpy_constrained_variables(m, dims*, ndims, set) -> void*` | Add the array of variables of dimensions `dims` constrained to belong to a vector set, as `@variable(model, [1:m, 1:n] in set)` |
| `jumpy_constant(m, value) -> void*` | a `Float64` node |
| `jumpy_integer_constant(m, value) -> void*` | an `Int64` node, preserving integer exponents and indices |
| `jumpy_apply(m, op, args**, nargs) -> void*` | Apply the native Julia operator to JuMP/GenOpt arguments |
| `jumpy_iterator(m, values*, len) -> void*` | `GenOpt.iterator` over `Float64` values |
| `jumpy_integer_iterator(m, values*, len) -> void*` | `GenOpt.iterator` over `Int64` values, including integer index ranges |
| `jumpy_float_array(m, values*, len) -> void*` | a data vector, 1-based-indexable in templates |
| `jumpy_array(m, integer, data*, rows, columns, ndims) -> void*` | a data vector or column-major matrix of `Int64` or `Float64` elements |
| `jumpy_getindex(m, array, first*, last*, n) -> void*` | `array[i, j]`, with a 1-based integer (`last` is `typemin(Int64)`) or range `first:last` per dimension |
| `jumpy_vcat(m, args**, nargs) -> void*` | The vector of affine expressions concatenating scalars and vectors, as `[a; b]` |
| `jumpy_nonlinear(m, head, args**, nargs) -> void*` | The nonlinear expression `head(args...)`, as a call of a `JuMP.NonlinearOperator` |
| `jumpy_new_set(index, args**, sizes*, nargs) -> uint64` | Construct and retain the set of the registry `index` from pointers to the data of its fields and their numbers of rows and columns |
| `jumpy_free_set(id) -> int32` | Release a native set handle |
| `jumpy_add_constraint(m, f, id) -> void*` | Add a scalar or vector constraint or generated group and return its rooted reference; NULL on error |
| `jumpy_set_objective_sense(m, sense) -> int32` | Set the JuMP objective sense; 0 = min, 1 = max |
| `jumpy_set_objective_function(m, f) -> int32` | Set the native JuMP objective expression |
| `jumpy_optimize(m) -> int32` | Optimize the raw MOI backend; returns the termination status (`OPTIMAL == 1`) |
| `jumpy_set_time_limit_sec(m, seconds) -> int32` | Set a finite nonnegative time limit; NaN clears it; 0 on success, -1 on error |
| `jumpy_set_string_attribute(m, name, value) -> int32` | `JuMP.set_attribute(model, name, value)` with a `String` value; 0 on success, -1 on error |
| `jumpy_set_bool_attribute(m, name, value) -> int32` | Same with a `Bool` value, passed as `int32` |
| `jumpy_set_int_attribute(m, name, value) -> int32` | Same with an `Int64` value |
| `jumpy_set_float_attribute(m, name, value) -> int32` | Same with a `Float64` value |
| `jumpy_value(m, expr) -> float64` | Evaluate the native expression using MOI variable values; NaN on error |
| `jumpy_objective_value(m) -> float64` | Query the solver objective value; NaN on error |
| `jumpy_has_values(m) -> int32` | Query primal result availability; 0 or 1, -1 on error |
| `jumpy_has_duals(m) -> int32` | Query dual result availability; 0 or 1, -1 on error |
| `jumpy_termination_status(m) -> int32` | Query the MOI termination status without solving; -1 on error |
| `jumpy_dual(m, constraint) -> float64` | Query a scalar constraint's raw MOI dual; NaN on error |

Arrays and their indexed variables are ordinary native expression references.
Python translates local zero-based indices to Julia's one-based `getindex`
through `jumpy_apply`; no global column IDs or variable reconstruction are
needed. Value queries use the expression reference directly, with no Python
solution-vector cache. Constraint references are rooted with the model, just
like expression nodes, and become invalid when it is freed. Scalar dual queries
reject foreign references, expression pointers, and generated constraint groups.
They require a current dual solution and use MOI's raw sign convention.

`model.lazy_sum(template)` uses `jumpy_apply` with the `lazy_sum` operation to
create a native GenOpt `LazySum`. Its template and iterators remain intact for
structure-aware solvers such as ExaModels through JuliaCall. The compiled HiGHS
adapter calls GenOpt's polynomial lowering directly for affine lazy sums and
quadratic lazy-sum objectives. Python's built-in `sum()` remains available for
explicit sums.

The registry has one line per set, `set`, its 1-based index for
`jumpy_new_set`, its name, such as `MathOptInterface.LessThan` or
`MathOptVRP.TimeWindows{WITHOUT_START_TIME}` (type parameters are omitted, value
parameters are kept), its field names and its field types, such as
`Matrix{Float64}`; and one line per operator, `operator`, its index, its name,
such as `MathOptVRP.op_sum_distances`, and its head for `jumpy_nonlinear`.

Native sets use checked `uint64` identity handles, not set-kind codes; constructors
return `0` on error. Comparisons, variable bounds, and constraint groups all use
these native sets. A set can be reused across models in the same image and remains valid
after a model closes, until the set is released. The Python wrapper releases
sets automatically and also supports `close()` and context managers. Handles
cannot be passed to JuliaCall or another image.

Python exposes `model.set_time_limit_sec(seconds)` and
`model.unset_time_limit_sec()` for setting and clearing time limits. Both use
the `jumpy_set_time_limit_sec` transport in the compiled backend; NaN is
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

The build uses Julia 1.13.2, a C compiler, Python 3.11 or later, and the
[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) frontend:

```bash
julia --project=@juliac -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaC", version="0.3.12"))'
```

This integration requires the patched development branches of JuMP and GenOpt.
JuMP provides concrete model/reference types through `concrete_direct_model`;
GenOpt provides direct generator expansion and concrete-reference-compatible
symbolic indexing. The `[sources]` of [`Project.toml`](Project.toml) select
JuMP's `bl/juliac` and GenOpt's `bl/juliac-safe` branches, and recipes inherit
them through their dependency on JuMPy. JuliaCall needs the GenOpt changes, but
can use released JuMP because it does not construct a concrete direct model.

From the repository root, build a recipe with

```bash
python julia/build_backend.py julia/backends/highs
```

It instantiates the recipe environment, traces its `workload.jl` with
`--trace-dispatch` in ordinary Julia, generates the corresponding roots with
`JuMPy.TrimDispatch`, and compiles the module with JuliaC into
`julia/backends/highs/build/bundle`, where `import jumpy.highs` finds it in a
development checkout. The image is compiled for `JULIA_CPU_TARGET=generic`,
or for the target given with `--cpu-target`. Add `--trace script.jl` to trace
more scripts in the recipe environment, `--package src` to copy the bundle
into `src/jumpy/_backends/highs` for the `jumpy-opt` wheel, as CI does:

```bash
python julia/build_backend.py julia/backends/highs --package src
uv build --wheel
```

or `--wheel dist` to build the separate wheel `jumpy-opt-<name>` of another
backend.

Notes:

- `--jl-option handle-signals=no` is required because the library is loaded
  into a Python process; Julia's signal handlers would conflict with Python's.
- `--bundle` makes the output relocatable: `bundle/lib/` contains
  `libjumpy_highs.so` next to the Julia runtime libraries
  (`bundle/lib/julia/`), and `bundle/share/julia/artifacts/` contains the
  HiGHS_jll artifact with `libhighs.so`. This is the two-shared-library
  layout: `libjumpy_highs.so` (our entry points + Julia runtime image) loads
  `libhighs.so` (the solver distributed by HiGHS_jll) dynamically.
- Trimming uses `--trim=unsafe-warn` with the explicit roots of
  [`JuMPy.add_trim_roots`](src/trim_roots.jl), the traced dispatch roots, and
  the roots of the recipe. Compiler warnings remain: this is not a
  `--trim=safe` guarantee. Validate the compiled artifact with the full
  Python tests and executable tutorials.
- The traced roots depend on the Julia/dependency versions; they are
  regenerated by each build. The image-build entry script does not execute a
  solver workload because JLL initialization has not run at that point.
- Functions of JLL packages are compiled with `infer = false`, so the image
  cannot call them; read their paths instead, as Vroom.jl does for the
  executable of vroom_jll.

Wheels cannot store symlinks, so [`package_bundle.py`](package_bundle.py) keeps
each library under its install name and the aliases that the bundled binaries
refer to, such as the `lib/libhighs.so` path of HiGHS, or `libpcre2-8`, which
Julia opens as `libpcre2-8.so`. Every push builds and tests the wheels for Linux
x86_64, macOS arm64 and Windows and uploads them as workflow artifacts.
Publishing a GitHub release also uploads them to PyPI as `jumpy-opt`.

## Testing

In-process tests of the entry points (fast, no compilation):

```bash
julia --project=julia/test julia/test/runtests.jl
```

End-to-end through the compiled library and Python ctypes:

```bash
JUMPY_BACKEND=juliac uv run --group tests pytest
```

The Python loader uses `$JUMPY_LIB` if it names `libjumpy_<name>`, else the
`_backends/<name>/lib/` directory of each `jumpy` package directory, else the
development bundle `julia/backends/<name>/build/bundle/lib/`.

## JuliaCall

For JuliaCall, use a PythonCall environment with the same patched GenOpt
branch added to it. For an existing environment:

```bash
export JUMPY_PYTHONCALL_PROJECT=/path/to/pythoncall/environment
julia --project="$JUMPY_PYTHONCALL_PROJECT" -e '
    using Pkg
    Pkg.add(url="https://github.com/blegat/GenOpt.jl.git", rev="bl/juliac-safe")
'
export PYTHON_JULIACALL_PROJECT="$JUMPY_PYTHONCALL_PROJECT"
export PYTHON_JULIACALL_EXE="$(julia -e 'print(joinpath(Sys.BINDIR, Base.julia_exename()))')"
```

Install solver packages in that same JuliaCall environment when needed, for
example `Pkg.add("HiGHS")`. Then import the solver through JuliaCall and pass
its constructor to `jp.Model`, or an instance to `jp.direct_model`.

## NexOR client

[`backends/nexor`](backends/nexor) uses the same native ABI as HiGHS.
Its direct JuMP model uses NexOR.jl's incremental optimizer, which serializes
the model for remote execution. Solver packages stay on the server. The server
accepts any set, so the recipe lists the sets of the C ABI: `jumpy.nexor`
exports the common modeling API and the scalar MOI sets of `jumpy.highs`,
including the time-limit, termination-status and result queries. Its workload
answers the solve with a mock transport. Build it with

```sh
python julia/build_backend.py julia/backends/nexor
```

Select it with `import jumpy.nexor as jp`, then `jp.Model()`. Models read
`NEXOR_API_KEY` and `NEXOR_SERVER_URL` (default `https://api.nexoropt.com`) when
created. There is no default remote solver: select it with
`model.set_attribute("solver", "HiGHS")`. Use one runtime per Python process.

`julia --project=julia/test julia/test/nexor.jl` exercises the native ABI and
MOF round trip with a mock transport, without contacting production. After
building, run `JUMPY_BACKEND=nexor uv run --group tests pytest
tests/test_nexor.py` to test the compiled client against a local HTTP fixture.
