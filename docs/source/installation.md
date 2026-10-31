# Installation

This branch's native JuMP/GenOpt integration is a prototype. Both building the
compiled backend and using JuliaCall currently require the patched development
checkout of GenOpt; released GenOpt 0.2.3 lacks the required lazy-sum bridge.
Building the compiled backend also requires the concrete-model JuMP development
patch. Follow the
[development environment setup](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md#building)
to select that checkout without changing the repository manifest.

Install JuMPy from PyPI. The distribution is named `jumpy-opt` since `jumpy`
is taken on PyPI; the package is still imported as `jumpy`:

```console
pip install jumpy-opt
```

Select the compiled HiGHS interface explicitly:

```python
import jumpy.highs as jp

model = jp.Model()
```

It uses the compiled backend shipped in the wheels for Linux x86_64 (glibc 2.17
or newer) and macOS arm64, and requires neither a Julia installation nor
`juliacall`. On other platforms, pip installs JuMPy without the compiled backend:
use the JuliaCall interface, or
[build the backend](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md#building)
and point `JUMPY_LIB` to it.

## JuliaCall interface

```console
pip install 'jumpy-opt[juliacall]'
```

```python
import jumpy.juliacall as jp

model = jp.Model()  # A JuMP model without a solver
```

Install and load a solver in the JuliaCall environment explicitly. For example,
with HiGHS:

```python
from juliacall import Main as jl

jl.seval('using Pkg; Pkg.add("HiGHS"); import HiGHS')
model = jp.Model(jl.HiGHS.Optimizer)
```

This forwards the optimizer factory to `JuMP.Model`. To use an existing optimizer
instance with JuMP's direct mode:

```python
model = jp.direct_model(jl.HiGHS.Optimizer())
```

This interface loads Julia and uses the same native JuMP/GenOpt expression
operations as the compiled backend. JuliaCall's `Model` uses normal JuMP
caching and GenOpt bridges. Its `direct_model` uses the optimizer without added
layers. Compiled HiGHS uses a concrete direct model over raw MOI, without
bridges. `jp.MOI` is Julia's actual MathOptInterface module. It
does not need the compiled JuMPy library. Both interfaces expose the same
modeling helpers and `jp.MOI`, but they are alternatives: choose one per Python
process, and do not pass models, expressions, or sets between them.

`Model()` takes no backend argument; the import selects it. The root package
no longer exports `jumpy.Model`.

## Development documentation

The executable tutorials require a locally built JuMPy library. Follow the
[compiled backend build instructions](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md),
then build the documentation with:

```console
uv run --group docs sphinx-build -M html docs/source docs/build -W --keep-going
```

Set `JUMPY_LIB` to the compiled library if it is not in the default development
location.

The tutorials import `jumpy.highs` and do not require JuliaCall.
