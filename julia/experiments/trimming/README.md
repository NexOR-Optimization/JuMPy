# Safe-trimming probes

These are historical isolated compiler reproducers, not an alternative JuMPy
backend. The normal backend now uses a bridge-free concrete JuMP model and
`--trim=unsafe-warn` with explicit roots; see [the backend README](../../README.md).
The solver probes expose only the fixed
test problems shown in their source; their C entry points are not a general,
exception-safe application interface.

Use the development environment described in [the backend README](../../README.md),
including the patched GenOpt checkout. From the repository root, first check
the problems in ordinary Julia, in separate processes:

```bash
julia --project="$JUMPY_JULIA_PROJECT" -e 'include("julia/experiments/trimming/raw_highs.jl"); @assert raw_highs_lp(2.0) ≈ 2.0'
julia --project="$JUMPY_JULIA_PROJECT" -e 'include("julia/experiments/trimming/static_lp.jl"); @assert static_lp(2.0) ≈ 2.0'
julia --project="$JUMPY_JULIA_PROJECT" -e 'include("julia/experiments/trimming/static_group.jl"); @assert static_group(2.0) ≈ 2.0'
```

Then compile one probe at a time. Safe trimming must pass verification; do not
substitute `unsafe-warn` to make a failed probe appear successful.

```bash
export JUMPY_TRIM_OUTPUT="$(mktemp -d)"
julia --project=@juliac -m JuliaC \
    --output-lib "$JUMPY_TRIM_OUTPUT/libprobe" \
    --project "$JUMPY_JULIA_PROJECT" \
    --compile-ccallable --trim=safe --jl-option handle-signals=no \
    julia/experiments/trimming/raw_highs.jl
```

Replace `raw_highs.jl` with `static_lp.jl`, `static_group.jl`, or `backend.jl`.
The last one roots JuMPy's full C ABI, without executing its build workload.
Workload execution is not proof of trimming coverage: image generation does
not normally run JLL initializers, and error-returning C wrappers can conceal
failed workload operations. The probes above separate normal execution from
compile-time reachability verification.

## Full native-handle backend with patched dependencies

With Julia 1.13.0 and JuliaC 0.3.10, the bridge-free standalone JuMP LP now
passes `--trim=safe` using the `bl/juliac` changes in JuMP, HiGHS,
MathOptInterface, and OrderedCollections. Its executable solves with both a
`Nothing` optimize hook and a concrete closure hook.

The full JuMPy C ABI does **not** yet pass. The initial `backend.jl` probe using
those patches, the concrete `ConcreteModel{Float64,HiGHS.Optimizer,Nothing}` alias,
and the GenOpt integration changes reported 12 verifier errors and zero verifier
warnings at these type-erased boundaries:

- `jumpy_apply`: a `Function`-typed operator applied to `Vector{Any}` arguments.
- `jumpy_add_constraint` and `jumpy_set_objective_function`: the expression
  obtained from its opaque handle has type `Any`.
- `jumpy_value`: the same unboxing issue, followed by conversion of an unknown
  return type to `Cdouble`.

This diagnostic deliberately omits the historical dispatch roots. Additional
roots retain methods but do not resolve these unknown calls for the safe
verifier. Conversely, this error count does not bound the remaining work:
resolving the wrapper calls can expose further errors inside their callees.

The temporary GenOpt integration also needs model-aware `moi_function` for
`ExprGenerator` and `AbstractModel` annotations on its ownership checks to
coexist with JuMP's new fused conversion interface. Those compatibility changes
were tested in the isolated checkout; they were not applied to GenOpt `main`.

The wrapper now checks the supported native types and operators in Julia;
Python retains the same opaque-handle API. Its 150 focused ordinary-Julia tests
pass, including calls through the native entry points and generated constraints
such as `(i + 1) * x[i + 1]`. This is runtime validation, not proof that the full
wrapper trims: the binary dispatcher still has an unresolved call, and exposing
more callees reveals further errors in GenOpt and nonlinear evaluation.
The latest full probe reports 349 verifier errors and zero warnings; as above,
these counts are not a measure of distance to completion.

GenOpt's `bl/juliac` worktree is based on refreshed `main` (`0fd23b0`). It adds
typed iterator handles, callable specialization, and a fix that keeps numeric
index templates from overriding the model's variable-reference type. Its
`prepare` method passes constructor arguments individually; JuMP's `bl/juliac`
also specializes the nonlinear constructor on the number of arguments. The
focused constructor executable built from these actual worktrees passes safe
verification and runs successfully. All 136 operator assertions and 26,053 JuMP
nonlinear assertions pass too.

This does not yet type the underlying shared iterator storage. A standalone
group probe in the integration environment still reports 18 verifier errors,
starting from mapping data through `Iterator.values::Vector` and propagating
to expression construction. Nonlinear tree traversal and operator-registry
callbacks are additional remaining boundaries in the full backend. No safe
full-backend shared library was produced, so compiled Python tests and tutorials
could not be run against a safe artifact.

## Initial results

Tested with Julia 1.13.0, JuliaC 0.3.10, HiGHS.jl 1.24.1, and the local GenOpt
development checkout. Error counts refer to separate entry-point sets; they
are not additive or a measure of distance to completion.

| Probe | Normal Julia | Safe-trim verifier errors |
|---|---|---|
| Raw HiGHS/MOI LP | Pass | 9 |
| Static bridge stack, scalar LP | Pass | 17 |
| Static bridge stack, one GenOpt group | Pass | 35 |
| Earlier bridge-based JuMPy C ABI | Passed normal backend tests | 244 (plus 2 warnings) |

The failures distinguish several independent boundaries:

- **JuMP:** the backend field and the model stored in a variable reference both
  need to preserve concrete types. Changing only the model is insufficient.
- **JuMPy's C interface:** `Any`-boxed objects and a runtime operator name need
  checked dispatch to the finite native types/operators supported by the image.
  This belongs in Julia; Python can keep passing opaque references. Explicit
  entry-point roots alone do not resolve arbitrary dynamic dispatch.
- **MOI bridges:** a static `SingleBridgeOptimizer` stack avoids dynamic bridge
  selection, but its bridge maps/finalization still erase types.
- **GenOpt:** group expansion has abstract iterator domains, runtime-dimensional
  indexing, and `Any`-typed nonlinear expression evaluation.
- **HiGHS:** option replay reads `Dict{String,Any}`, and the default interrupt
  callback calls a `Function`-typed field. Even the bridge-free LP encounters
  these paths.
- **Diagnostics:** dictionary key-conversion errors reach MOI/JuMP printing
  methods that the verifier cannot resolve. Successful numerical paths do not
  make these reachable error paths safe to omit.

These probes do not establish that the complete native-handle backend passes
safe verification. They are not requirements for the current bridge-free,
warning-mode build: that backend bypasses dynamic JuMP solver wrappers via
`JuMP.backend(model)` and calls GenOpt's direct expansion helper.

## Successful reduced slices

Two isolated slices did pass `--trim=safe` and execute in fresh Python processes
after temporary dependency changes. These changes are not in the repository's
dependency manifest, and the results do not imply that the full backend trims.

| Slice | Required prototype changes | Fresh-process checks |
|---|---|---|
| Native opaque-handle arithmetic | Concrete JuMP model/reference types; typed `UnorderedPair` display | All 32 `+`/`*` input combinations; affine/quadratic results; clean unsupported-call errors and reuse afterward |
| Raw HiGHS/MOI LP | Normalized concrete option storage; separate concrete default interrupt callback; typed MOI variable display | Four LP solves with distinct right-hand sides |

`native_arithmetic.jl` retains actual Julia objects behind opaque pointers and
dispatches to actual Base/JuMP operators. Its finite checks are all in Julia;
there is no Python expression classification. It deliberately creates its
in-memory model at build time and does not claim to trim runtime model
construction, solving, or GenOpt groups.

`patched_raw_highs.jl` requires the temporary HiGHS patch and explicitly loads
the experimental MOI display specialization. The HiGHS patch normalizes option
values to `Union{Bool,Int64,Float64,String}` and separates the built-in interrupt
callback from user callbacks without disabling interruption. Twenty-seven
existing HiGHS option/user-callback checks passed in ordinary Julia.

After compiling the respective source with the command above and its patched
dependency environment, run these scripts in separate fresh processes:

```bash
python julia/experiments/trimming/smoke_native_arithmetic.py /path/to/libarithmetic.so
python julia/experiments/trimming/smoke_raw_highs.py /path/to/libraw_highs.so
```

The raw solver artifact was tested against the installed Julia runtime and
cached JLL dependencies, not as a portable bundle.

## Concrete JuMP prototype

`concrete_jump_tests.jl` is an ordinary Julia regression test for a temporary
JuMP prototype that preserves the existing `Model`, `VariableRef`, `AffExpr`,
and `QuadExpr` aliases and opts into concrete backend/reference types through
`JuMP.concrete_direct_model`. That constructor is **not** a released JuMP API.
Run this test only in an environment developed against the prototype.

Its 28 assertions cover inferred model/reference types, native
affine/quadratic/nonlinear expression types, MOI conversion round trips,
LP/QP solves, and the existing aliases. These runtime tests do not assert
that the model's entire reachable code passes safe trimming.

The temporary JuMP patch also passed 24,377 assertions from the existing model,
variable, expression, operator, and objective suites (including Float32 and
custom-extension variants), with zero detected method ambiguities. This is
representative coverage, not the entire JuMP test suite. The prototype's new
concrete models do not match extension methods restricted to the old fixed-backend
`GenericModel` alias; extensions must generalize those signatures too.

A C-entry LP using the concrete model still failed safe verification. Beyond
the downstream failures above, it exposed JuMP's unspecialized result-forwarding
varargs, `Any`-typed optimize hook, error rethrow dispatch, and nonlinear-interface
checks through dynamically typed constraint lists. These need separate treatment;
the backend/reference type parameters alone do not complete the JuMP work.
