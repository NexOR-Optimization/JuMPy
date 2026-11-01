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

## Transposed constraint matrix experiment

`matrix_highs.jl` adds a `MatrixOfConstraints` storing the linear rows in
`Transpose{Float64,SparseMatrixCSC{Float64,HiGHS.HighsInt}}`. Wrapping the
hardcoded matrix shares its coefficient vector. The experimental loader
converts the one-based indices for HiGHS while preserving the source matrix.

```sh
julia --project=benchmarks/paper benchmarks/paper/matrix_highs.jl 1000
```

It compares the direct baseline, matrix wrapping, eager `MOI.add_constraints`
into the matrix, and HiGHS's native `addRows` with one call or batches of
65,536 rows. The two function-based paths include creating the same scalar
MOI functions. Every total includes initial data assembly, insertion, final
solver loading, and the zero-limit HiGHS run; disposal is excluded.

This prototype handles scalar equality rows and fixed variable numbering.
Objective and variable-bound arrays come from `direct_highs.jl`. The custom
loader targets a raw HiGHS handle; full optimizer attribute and result
bookkeeping is outside this experiment. See the
[timings and independent-process check](results/matrix-highs-2026-11-01/table.md).

## GenOpt to native row batches

`native_genopt.jl` continues from native `addRows`, using GenOpt's real
template planner and the LQCP formulation in `grouped_models.jl`:

```sh
julia --project=benchmarks/paper benchmarks/paper/native_genopt.jl 1000
```

The scalar-function path uses GenOpt's typed fill kernels, then packs the
functions for native insertion. The column path fills native row buffers
directly and merges duplicate entries without per-row function objects.
The local adapter specializes GenOpt's private insertion hook; it supports
only affine equality groups and does not change the installed packages.

The `native-*` and `genopt-*` paths retain the direct builder's variable bounds
and objective. `genopt-mapped` adds HiGHS.jl variable mapping and row metadata.
The `grouped-*` paths build the complete grouped JuMP model, including
variables, bounds, objective, and `MOI.optimize!` result storage. `grouped-stock`
uses the installed packages without the experimental adapter.
`grouped-bounds` batches bound updates, and `grouped-bulk` also creates the
variable blocks with native `addCols`. Both retain ordinary JuMP variable
references and HiGHS.jl metadata.

Each path warms up at the requested size, then records five samples with
garbage collection before each sample. Compilation, pre-sample collection,
and model disposal are excluded. HiGHS uses one thread, no presolve, and a
zero solve time limit. See the [results and phase timings](results/native-genopt-2026-11-01/table.md).

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
