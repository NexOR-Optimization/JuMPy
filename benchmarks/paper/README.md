# JuMP paper benchmarks

Ports of [JuMPPaperBenchmarks at b4a3479](https://github.com/jump-dev/JuMPPaperBenchmarks/tree/b4a3479):
facility location, linear-quadratic optimal control (LQCP), and the small latency LP.
The Julia formulations are retained in `models.jl`; `models.py` expresses the
scaling cases with GenOpt iterators, constraint groups, and lazy sums.
Array bounds and integrality use bulk native calls instead of Python loops.
The facility assignment has an n-term Python loop to form one template, rather
than a Python loop over its n(n+1)² customer/facility constraints.

## Four-way HiGHS comparison

From the repository root, with Julia 1.13 and Python 3.10 or newer:

```sh
julia --project=benchmarks/paper -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/paper/build/tools -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaC", version="0.3.12"))'
python benchmarks/paper/build_native.py --juliac-project benchmarks/paper/build/tools
python benchmarks/paper/run.py \
  --library benchmarks/paper/build/highs/lib/libjumpy_highs.so \
  --output benchmarks/paper/results/my-highs-run
```

The default runs LQCP at n = 100, 250, 500, 1000 with three warmed samples per
size, in this order:

1. **JuMP cached**: ordinary `Model(HiGHS.Optimizer)`; includes copying the cache.
2. **JuMP direct**: `concrete_direct_model(HiGHS.Optimizer())`, ordinary JuMP macros.
3. **JuMP direct + GenOpt**: a direct model, bulk bounds, four constraint
   templates, and lazy sums, constructed entirely in Julia.
4. **JuMPy.HiGHS**: `import jumpy.highs as jp; jp.Model()`, expressing the same
   templates from Python through the compiled C ABI.

The third and fourth paths share JuMPy's direct adapter, raw HiGHS backend,
variable order, and GenOpt expansion code. `grouped_models.jl` and `models.py`
keep matching recipes, including array bounds and integrality. Their difference
is where templates are constructed and how they reach Julia. No cache or bridge
optimizer is inserted in either path. Ordinary JuMP uses its macros to build
expressions in place; string names are disabled throughout.

Each frontend runs in a separate process with one Julia/BLAS/solver thread.
Compilation and one n = 5 warm-up are excluded. Each sample starts after full
garbage collection; model disposal is excluded. Construction and `optimize!`
with presolve off and a zero time limit are timed, including solver ingestion.
These are **model-building timings**, not a comparison of solving speed.
JuliaC uses `--trim=no`, so this measures warmed compiled-library operation,
not cold startup or the eventual fully trimmed release image.

HiGHS cannot accept the facility model's quadratic constraints. To compare that
model, select `--backend storage` or `--backend gurobi` and build the matching
library. Those backends default to the paper's facility sizes 25, 50, 75, 100.

For a quick run, add `--lqcp-sizes 10 25 --repeats 2`. Use `--frontends` to select
a subset of `jump-cached jump-direct jump-genopt jumpy`. The runner writes:

- `samples.jsonl`: every warm-up and individual timing, with package versions.
- `summary.csv`: median, minimum, maximum, variable count, and peak process RSS.
- `metadata.json` and `Manifest.toml`: machine, threads, library hash, and environment.
- `table.md`: a Markdown table ready to paste into the root README.
- `charts.md`: Mermaid bar charts for the largest instance of each model.

The final column, **Python / grouped Julia**, divides the two medians. A ratio
near 1 supports the hypothesis that Python adds little overhead; it is measured,
not assumed. Small cases can expose fixed call costs, and solver ingestion can
dominate larger cases. Inspect the full sample range before interpreting small
differences. Peak RSS includes each runtime and is not a per-model allocation
measurement.

Regenerate the report from an existing CSV without rerunning any benchmark:

```sh
python benchmarks/paper/report.py benchmarks/paper/results/my-highs-run/summary.csv \
  --output benchmarks/paper/results/my-highs-run/table.md \
  --mermaid benchmarks/paper/results/my-highs-run/charts.md
```

GitHub renders fenced Mermaid diagrams in README files; the generated
`xychart-beta` snippets need no uploaded image. For a four-series scaling plot
with shaded sample ranges, SVG is clearer. Install `matplotlib`, then add
`--plot` to the runner or run:

```sh
python benchmarks/paper/plot.py benchmarks/paper/results/my-highs-run
```

This produces `comparison.svg` and `comparison.png`. Link the SVG from a README
with `![Model construction timings](benchmarks/paper/results/my-highs-run/comparison.svg)`.

## Historical storage results

The recorded run is **construction only**, on identical MOI storage, with three
warmed repetitions per case and one thread. Cached JuMP includes materializing
its cache into that storage. String names are disabled in all frontends.
The native C-ABI image uses JuliaC with `--trim=no`; its initial compilation is
excluded from these warm timings. These are not measurements of a fully trimmed
release image or of solver performance.

The original paper also calls Gurobi with a zero time limit and presolve disabled
to include solver ingestion. That exact mode is supported by the runners, but
Gurobi rejects this sandbox's host ID (zero), so it was not measured here.

Median construction times for the largest instances:

| Model | Variables | JuMP direct | JuMPy baseline | JuMPy with prepared GenOpt expansion | JuMP cached |
| --- | ---: | ---: | ---: | ---: | ---: |
| Facility, n = 100 | 4,080,601 | 4.97 s | 7.00 s | 1.70 s | 9.81 s |
| LQCP, n = 2000 | 4,006,001 | 13.58 s | 13.69 s | 2.00 s | 19.63 s |

Preparing GenOpt's coefficient and index columns before expanding the rows
reduces JuMPy construction time by about 4.1 times for facility location and
6.8 times for LQCP. These measurements use a rebuilt JuliaC library containing
the change from GenOpt's `bl/batch-expansion` branch. The
[comparison samples](results/batch-expansion-2026-10-31/complete-models.csv)
and [build metadata](results/batch-expansion-2026-10-31/metadata.json) record the
package versions, compilation settings, and coefficient checks.

The plot below shows the baseline across all recorded sizes. Prepared expansion
was measured on the two largest instances; all results are three-sample
measurements on one machine.

![Baseline construction timings](results/storage-2026-10-31/comparison.png)

See [all timing samples](results/storage-2026-10-31/samples.jsonl),
[summary CSV](results/storage-2026-10-31/summary.csv),
[machine and methodology](results/storage-2026-10-31/metadata.json),
[package versions and source hashes](results/storage-2026-10-31/versions.json),
and [coefficient-level validation](results/storage-2026-10-31/validation.json).
The machine was an Intel i7-12700K, with Julia 1.13.3 and JuMP 1.32.1.

The [generator expansion benchmark](generator_expansion.jl) isolates the facility
coordinate and LQCP PDE constraint families on the same MOI storage. Median
construction times and allocated bytes from three warmed runs were:

| Constraint family | Rows | Scalar | Prepared | Allocated bytes, scalar → prepared |
| --- | ---: | ---: | ---: | ---: |
| Facility coordinates, n = 100 | 1,020,100 | 6.58 s | 0.208 s | 11.60 GB → 0.53 GB |
| LQCP PDE, n = 2000 | 3,998,000 | 5.62 s | 1.82 s | 4.20 GB → 3.01 GB |

These are measurements of one constraint family at a time. The
[raw samples](results/batch-expansion-2026-10-31/expansion.csv) include GC time.
Prepared columns use additional temporary memory; output functions and term
vectors remain independent for every row.

The latency LP was also solved with HiGHS in five fresh processes per frontend,
each checked against objective 205. Median process wall times were **0.454 s**
for JuMPy, **2.326 s** for direct JuMP, and **2.954 s** for cached JuMP. These
compare JuMPy's prebuilt native image with ordinary Julia startup and installed
package caches. No custom JuMP sysimage was used, so this does not reproduce
the paper's custom-sysimage latency comparison. See the
[latency samples](results/latency-2026-10-31.json).

## Reproduce

From the repository root, with Julia 1.13 and Python 3.10 or newer:

```sh
julia --project=benchmarks/paper -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/paper/build/tools -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaC", version="0.3.12"))'
python benchmarks/paper/build_native.py --backend storage --juliac-project benchmarks/paper/build/tools
python benchmarks/paper/run.py --backend storage \
  --library benchmarks/paper/build/storage/lib/libjumpy_storage.so \
  --output benchmarks/paper/results/my-storage-run
```

The runners use the repository's Python source directly; installing a wheel is
not required. Set `--repeats`, `--facility-sizes`, or `--lqcp-sizes` for a smaller
run. For example, `--facility-sizes 5 10 --lqcp-sizes 50 100` is a quick check.
The recorded package environment uses the same JuMP/GenOpt/MOI JuliaC branches
as the compiled JuMPy backend. Source hashes are retained because these branches
can move independently of package version numbers.

The expansion-only benchmark uses private helpers from GenOpt 0.2.5. With that
version checked out in a sibling `GenOpt` directory and its test environment
instantiated, run from the JuMPy root:

```sh
JULIA_NUM_THREADS=1 julia --startup-file=no --project=../GenOpt/test \
  benchmarks/paper/generator_expansion.jl
```

It warms both expansion paths, alternates three samples, and prints construction
time, GC time, and allocated bytes as CSV. It does not solve the models.

To reproduce the original Gurobi timing scope on a host with a matching license:

```sh
python benchmarks/paper/build_native.py --backend gurobi --juliac-project benchmarks/paper/build/tools
python benchmarks/paper/run.py --backend gurobi \
  --library benchmarks/paper/build/gurobi/lib/libjumpy_gurobi.so \
  --output benchmarks/paper/results/my-gurobi-run
```

`--backend highs` can run LQCP and the latency LP; HiGHS does not support the
facility model's quadratic constraints. Build its library with
`build_native.py --backend highs`. Linux shared-library paths above use `.so`;
use the corresponding `.dylib` path on macOS.

To measure the latency LP with HiGHS:

```sh
python benchmarks/paper/build_native.py --backend highs --juliac-project benchmarks/paper/build/tools
python benchmarks/paper/latency.py \
  --library benchmarks/paper/build/highs/lib/libjumpy_highs.so \
  --output benchmarks/paper/results/my-latency-run.json
```

## Validate a port

Export a small reference and candidate without changing the timed large runs:

```sh
julia --project=benchmarks/paper benchmarks/paper/run_jump.jl \
  --model facility --sizes 3 --repeats 1 --mode direct \
  --export /tmp/reference.mof.json
python benchmarks/paper/run_jumpy.py --backend storage \
  --library benchmarks/paper/build/storage/lib/libjumpy_storage.so \
  --model facility --sizes 3 --repeats 1 --export /tmp/candidate.mof.json
python benchmarks/paper/validate.py /tmp/reference.mof.json /tmp/candidate.mof.json
```

Repeat with `--model lqcp`. The comparison canonicalizes row order, objective
representation, variable names, and floating-point coefficients to 12 decimal
places, then checks every constraint and objective coefficient. The recorded
small facility model has 199 variables and 316 constraints; LQCP has 19 and 54.
The ports follow the paper's Julia code, including `M = 2√2` and the Julia LQCP
objective parentheses, rather than the different Pyomo expressions.

`python benchmarks/paper/plot.py <result-directory>` produces PNG and SVG
figures and requires matplotlib. Benchmark execution itself uses Python's
standard library only.
