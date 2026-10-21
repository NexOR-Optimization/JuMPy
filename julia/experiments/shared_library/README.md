# Shared constructor library experiment

Historical probes behind the production integration. For the supported
one-image build and automatic startup, see [the backend README](../../README.md).
The standalone/two-image probes below remain experiments; production uses
`julia/build.jl` and is tested by `tests/test_shared_image.py`.

This is a standalone experiment, not a replacement for either JuMPy backend.
`constructors.jl` compiles the shared `JuMPyMOI` source into a constructor-only
JuliaC library. It exposes a small C ABI for `LessThan`, variables, constants,
and addition expressions. It does not import HiGHS or PythonCall.

The library roots real MOI objects in a registry. Its `UInt64` handles belong
only to that registry: they are not Julia object pointers or portable object
identities. A diagnostic API exposes datatype addresses for comparison only.
The probes never dereference an address using a foreign Julia runtime.

## Observed results

Tested on Linux on 2026-09-19 with Julia 1.13.0, JuliaC 0.3.10, and
MOI 1.53.0 in the constructor image and JuliaCall environment:

| Setup | Result |
| --- | --- |
| Constructor `.so` alone | Construction, GC, release, and invalid-handle checks pass. |
| Constructor `.so` as JuliaCall's system image | Same MOI types; native set/function objects solve the HiGHS example with optimum 3. |
| Constructor `.so` loaded after ordinary JuliaCall | Second module remains unregistered; constructor exports deliberately not called. |
| Constructor `.so` loaded after compiled `libjumpy_highs` | Same second-image limitation. |
| Privatized constructor `.so` alongside JuliaCall | Both runtimes work through primitive accessors, but all four datatype identities differ. |

The complete 26-test Python suite also passes with the constructor image as
JuliaCall's system image. Separately, the shared-source refactor passes all
129 Julia assertions and all 26 Python tests against the compiled HiGHS backend.

Thus one constructor image can host JuliaCall and dynamically loaded Julia
packages. These results do **not** establish live-object sharing between two
independently compiled JuliaC images, or compatibility after trimming.

## One image containing constructors and HiGHS

`combined.jl` includes the real `JuMPyHiGHS` C entry points and the constructor
exports in **one** JuliaC image. Both refer to the same `JuMPyMOI` module.
An experimental `jumpy_probe_add_constraint` entry point consumes the native
`LessThan` from the shared registry directly; it does not reconstruct the set
from a Python definition, a sense tag, or a serialized value.

This configuration was also built and tested successfully with Julia 1.13.0
and JuliaC 0.3.10:

- The compiled library creates `LessThan(1.0)` and solves `max x + y` subject
  to `x + y <= 1`, `x, y >= 0`, without importing JuliaCall. This also passes
  with Python's site packages disabled (`python -S`).
- JuliaCall can initialize from that image first, or attach **after** the C
  constructor has already created the set. The same native set is accepted
  by both backends in the same process, with optimum 1 in each model.
- A JuliaCall-created `MOI.LessThan(2.0)` can also be rooted in the common
  registry and consumed by the compiled backend, with optimum 2.
- JuliaCall's borrowed set remains usable after releasing its registry handle,
  clearing the consumer's diagnostic reference, and running GC.
- Invalid, released, and wrong-model handles return errors. An already active
  default JuliaCall image is rejected before invoking compiled exports.
- The existing 26-test Python suite passes with each backend using this image.

### Reproduce the combined test

After setting up `JUMPY_SO_WORK`, the runtime environment, and the Python
environment as below, run these commands from the repository root:

```bash
julia --project="$JUMPY_SO_WORK/runtime" -e 'using Pkg; Pkg.develop(path=joinpath(pwd(), "julia")); Pkg.add("GenOpt")'

julia --project="$JUMPY_SO_WORK/toolchain" -m JuliaC \
    --output-lib "$JUMPY_SO_WORK/libjumpy_highs_moi" \
    --project "$JUMPY_SO_WORK/runtime" \
    --compile-ccallable --jl-option handle-signals=no \
    julia/experiments/shared_library/combined.jl

uv run --extra juliacall python julia/experiments/shared_library/combined_probe.py \
    all "$JUMPY_SO_WORK/libjumpy_highs_moi.so"
```

`combined_probe.py` runs four child processes: compiled-only, JuliaCall-first,
compiled-first, and rejection of an incompatible active image. For shared use
it selects the `.so` as `PYTHON_JULIACALL_SYSIMAGE` before initialization. For
compiled-first startup it also sets `JULIA_PROJECT` to the selected JuliaCall
environment **before** calling `jl_init_with_image_handle`. Without this,
JuliaCall can attach to the runtime but fails to find `PythonCall` in its
already-selected package environment. This startup configuration is required;
sharing a filename alone is not sufficient.

This file remains an untrimmed experimental adapter. Automatic shared-runtime
startup is now integrated into the production backend; the public
`Model.constraint(func, set)` API is still deferred.

## Build and run

Run from the repository root with Julia 1.13 and a Python environment containing
JuliaCall. The commands below keep dependencies and artifacts in a temporary
directory, without changing the repository's Julia manifest. Using the same
Julia version and MOI dependency resolution for the image and JuliaCall matters.

```bash
JUMPY_SO_WORK=$(mktemp -d /tmp/jumpy-shared-so.XXXXXX)
export JULIA_DEPOT_PATH="$JUMPY_SO_WORK/depot:${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_NUM_PRECOMPILE_TASKS=2

julia --project="$JUMPY_SO_WORK/toolchain" -e 'using Pkg; Pkg.add("JuliaC")'
julia --project="$JUMPY_SO_WORK/runtime" -e 'using Pkg; Pkg.add(["MathOptInterface", "HiGHS", "PythonCall", "OpenSSL_jll"])'

julia --project="$JUMPY_SO_WORK/toolchain" -m JuliaC \
    --output-lib "$JUMPY_SO_WORK/libjumpy_moi" \
    --project "$JUMPY_SO_WORK/runtime" \
    --compile-ccallable --jl-option handle-signals=no \
    julia/experiments/shared_library/constructors.jl

export PYTHON_JULIAPKG_PROJECT="$JUMPY_SO_WORK/runtime"
export PYTHON_JULIAPKG_EXE="$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, Base.julia_exename()))')"
uv run --extra juliacall python julia/experiments/shared_library/probe.py \
    all "$JUMPY_SO_WORK/libjumpy_moi.so"
```

Each probe runs in a fresh child process. `all` checks:

1. **Standalone:** construct objects through the C ABI, force GC, and release
   handles. Invalid handles return sentinels instead of dereferencing memory.
2. **Shared system image:** start JuliaCall with this `.so` as
   `PYTHON_JULIACALL_SYSIMAGE`. Check exact MOI datatype identity, retrieve the
   actual objects from their owning Julia module, and solve `max x` subject to
   `x + 1 <= 4` with HiGHS loaded as an ordinary Julia package. This is **one
   runtime and one image**, with HiGHS loaded dynamically, not two JuliaC images.
3. **Second image after JuliaCall:** start ordinary JuliaCall, load the `.so`,
   call `jl_init_with_image_handle`, and check whether its module registered.
   Do not call constructor exports from an unregistered image.

To check the same second-image boundary after a compiled JuMPy backend instead,
build `libjumpy_highs` as described in [the backend README](../../README.md) and
add `--backend-library /absolute/path/libjumpy_highs.so` to `all`. The
`second-compiled-image` mode verifies that the backend's module is registered
and checks the constructor module through the backend's own runtime API.

The second-image checks report `expected limitation confirmed` when the second
module is absent. That is a successful diagnostic, **not successful composition**.

## Privatization

Build a separate bundled variant on Linux:

```bash
julia --project="$JUMPY_SO_WORK/toolchain" -m JuliaC \
    --output-lib "$JUMPY_SO_WORK/libjumpy_moi_private" \
    --project "$JUMPY_SO_WORK/runtime" \
    --compile-ccallable --jl-option handle-signals=no \
    --bundle "$JUMPY_SO_WORK/private" --privatize=jumpymoi \
    julia/experiments/shared_library/constructors.jl

uv run --extra juliacall python julia/experiments/shared_library/probe.py \
    private "$JUMPY_SO_WORK/private/lib/libjumpy_moi_private.so"
```

The private probe first verifies that JuliaCall and the library resolved
different runtime functions. It then checks that primitive constructor/accessor
calls work, while the MOI datatype identities differ and the private module
remains absent from JuliaCall. It never transfers live objects between runtimes.

Trimming is deliberately disabled: this experiment isolates image/runtime
composition from the separate problem of a fixed, trim-safe callable ABI.
