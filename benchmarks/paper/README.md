# JuMP paper benchmarks

Compare four ways to build the
[JuMP paper models](https://github.com/jump-dev/JuMPPaperBenchmarks/tree/b4a3479):

1. **JuMP cached**: `Model(optimizer)`.
2. **JuMP direct**: `concrete_direct_model(optimizer())` with JuMP macros.
3. **JuMP direct + GenOpt**: templates and lazy sums built in Julia.
4. **JuMPy**: the same templates built in Python.

Choose `--backend highs` or `--backend nexor`. HiGHS includes solver ingestion
with presolve disabled and a zero solve time limit. NexOR measures local model
construction, including cache materialization for cached JuMP. It never calls
`optimize()` and needs no server, API key, or remote solver.

String names are disabled and Julia, BLAS, and HiGHS use one thread.
Compilation and warm-up are excluded. Native garbage collection runs normally
in both Julia and JuMPy. Model disposal is outside the construction timer.

## Run

From the repository root, prepare the Python and Julia environments:

```sh
python -m pip install -e .
julia --project=benchmarks/paper -e 'using Pkg; Pkg.instantiate()'
```

Use the normal compiled [HiGHS](../../julia/README.md#building) or
[NexOR](../../julia/README.md#nexor-client) backend, built with
`--trim=unsafe-warn`. For a local comparison, use the benchmark Julia environment
for that build too, so the Julia runners and compiled library use matching
dependencies. A HiGHS platform wheel built from this checkout also works.

```sh
python benchmarks/paper/run.py --backend highs \
  --output benchmarks/paper/results/highs

python benchmarks/paper/run.py --backend nexor --export-mof \
  --output benchmarks/paper/results/nexor
```

The normal package loader finds the library. Use `--library /path/to/library`
or `JUMPY_LIB` to select a normal local build explicitly.

LQCP defaults to n = 100, 250, 500, 1000, with three warmed samples each.
NexOR also runs facility at n = 25, 50, 75, 100; its quadratic constraints
require NexOR's local storage rather than HiGHS. For a quick local check:

```sh
python benchmarks/paper/run.py --backend nexor --facility-sizes 3 \
  --lqcp-sizes 3 5 --repeats 1 --export-mof \
  --output benchmarks/paper/results/quick-check
```

Use `--facility-sizes` without values to skip facility. Select a subset with
`--frontends jump-cached jump-direct jump-genopt jumpy`.

## Direct HiGHS experiment

`direct_highs.jl` builds LQCP's sparse arrays by hand and loads the complete
model through HiGHS.jl's C API wrappers. It isolates the cost of matrix
assembly and HiGHS itself, bypassing JuMP and MOI model storage.

```sh
julia --project=benchmarks/paper benchmarks/paper/direct_highs.jl 100 250 500 1000
```

It prints five warmed samples per size as JSON, separating array assembly,
model loading, and the zero-limit HiGHS run. Model disposal is excluded.
The [recorded timings](results/direct-highs-2026-11-01/table.md) show the
hardcoded Julia baseline.

## MOF export and reports

Both public backends support `model.write_mof("problem.mof.json")` without
solving. With NexOR, exporting is entirely local. No NexOR.jl change is needed.

`--export-mof` measures export separately after construction: copying to MOF,
JSON serialization, and writing a temporary local file. It generates
`mof-summary.csv` and `mof-table.md` alongside the construction results.

The runner saves every sample in `samples.jsonl`, construction statistics in
`summary.csv`, a README-ready `table.md`, and Mermaid bar charts in `charts.md`.
The **Python / grouped Julia** column divides the medians; above 1 means JuMPy
was slower. Regenerate a table from a CSV with:

```sh
python benchmarks/paper/report.py benchmarks/paper/results/nexor/summary.csv \
  --output benchmarks/paper/results/nexor/table.md
```

For SVG/PNG plots, install `matplotlib` and add `--plot` to the runner.

To keep an exported model for coefficient validation, run a frontend directly
with `--export path/to/model.mof.json`. Compare two exports using
`python benchmarks/paper/validate.py reference.mof.json candidate.mof.json`.
The comparison ignores names and row order and rounds coefficients to 12
decimal places.
