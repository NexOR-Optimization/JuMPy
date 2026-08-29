# Installation

Install JuMPy from PyPI:

```console
pip install jumpy
```

The default `juliac` backend uses the compiled HiGHS backend shipped with
JuMPy and does not require a Julia installation.

## Development documentation

The executable tutorials require a locally built JuMPy library. Follow the
[compiled backend build instructions](https://github.com/NexOR-Optimization/JuMPy/blob/main/julia/README.md),
then build the documentation with:

```console
uv run --group docs sphinx-build -M html docs/source docs/build -W --keep-going
```

Set `JUMPY_LIB` to the compiled library if it is not in the default development
location.
