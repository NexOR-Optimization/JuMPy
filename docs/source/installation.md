# Installation

This branch's native JuMP/GenOpt integration is a prototype. Both building the
compiled backend and using JuliaCall currently require the patched development
checkout of GenOpt; released GenOpt 0.2.1 is not sufficient. Follow the
[development environment setup](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md#building)
to select that checkout without changing the repository manifest.

Install JuMPy from PyPI:

```console
pip install jumpy
```

Select the compiled HiGHS interface explicitly:

```python
import jumpy.highs as jp

model = jp.Model()
```

It uses the compiled backend shipped with JuMPy and requires neither a Julia
installation nor `juliacall`.

## JuliaCall interface

```console
pip install 'jumpy[juliacall]'
```

```python
import jumpy.juliacall as jp

model = jp.Model()
```

This interface loads Julia and uses the same JuMP/GenOpt modeling source as the
compiled backend. `jp.MOI` is Julia's actual MathOptInterface module. It
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
