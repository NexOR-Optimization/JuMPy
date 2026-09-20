# Installation

Install JuMPy from PyPI:

```console
pip install jumpy
```

The default `juliac` backend uses the compiled HiGHS backend shipped with
JuMPy and requires neither a Julia installation nor `juliacall`.

## Optional JuliaCall interface

```console
pip install 'jumpy[juliacall]'
```

When the compiled library is available, JuMPy configures JuliaCall to use that
same shared image and Julia runtime. Its native MOI objects can then be used by
both interfaces, without Python copies of set definitions. JuliaCall needs a
Julia version matching the image and a compatible package environment containing
PythonCall; select these with `PYTHON_JULIAPKG_EXE` and
`PYTHON_JULIAPKG_PROJECT` when needed.

Start through JuMPy in a fresh Python process. An already initialized JuliaCall
using a different image is rejected; independent Julia images cannot exchange
live objects. If no compiled library is available, `backend="juliacall"` retains
its source-only fallback.

## Development documentation

The executable tutorials require a locally built JuMPy library. Follow the
[compiled backend build instructions](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md),
or build the default shared profile from the `julia/` directory:

```console
julia --project=@juliac -e 'using Pkg; Pkg.add("JuliaC")'
julia --project=@juliac build.jl shared
```

Then, from the repository root, build the documentation with:

```console
uv run --group docs sphinx-build -M html docs/source docs/build -W --keep-going
```

The default development location is `julia/build-shared/lib/`. Set `JUMPY_LIB`
to the library file for another location. Tutorials use compiled HiGHS and do
not import JuliaCall.

The explicitly selected `trimmed` build profile is experimental and
compiled-only. It cannot initialize JuliaCall/PythonCall, is not auto-discovered,
and uses `--trim=unsafe-warn`: its finite compiled method set does not guarantee
support for arbitrary models.
