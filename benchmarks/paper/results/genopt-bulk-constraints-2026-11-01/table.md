# GenOpt batch submission: LQCP

The [LQCP optimal-control model](../../grouped_models.jl), ported from
[JuMPPaperBenchmarks](https://github.com/jump-dev/JuMPPaperBenchmarks/tree/b4a3479),
built through JuMP direct + GenOpt. At n = 1000 it contains 1,003,001 variables
and 1,002,001 equality constraints, with a quadratic objective and variable bounds.

This comparison isolates the row-submission change in
[GenOpt PR #50](https://github.com/blegat/GenOpt.jl/pull/50). Times are seconds;
each entry is the median of three warmed samples. “Before” and “PR #50” use
stock HiGHS.jl; the last timing adds its native row-batch prototype.

| n | Variables | Before | PR #50 | PR #50 + native HiGHS rows | Before / native batch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 10,301 | 0.039 | 0.035 | 0.030 | 1.32× |
| 250 | 63,251 | 0.197 | 0.161 | 0.128 | 1.54× |
| 500 | 251,501 | 0.639 | 0.642 | 0.573 | 1.11× |
| 1000 | 1,003,001 | 2.913 | 2.919 | 2.295 | 1.27× |

At n = 1000, batching alone leaves the stock HiGHS scalar fallback essentially
unchanged (2.913 → 2.919 s). Providing the native row-batch method reduces the
complete construction time to 2.295 s, about 21% less time (1.27× speedup).
The later direct-buffer, bulk-bound and bulk-variable experiments are separate
changes and are excluded here.

One thread on an Intel i7-12700K; HiGHS presolve off and solve time limit zero.
Timings include variable creation, bounds, objective, generated rows, solver
ingestion and `MOI.optimize!` result storage. Compilation, imports and the n = 5
warm-up are excluded. Model disposal is not explicitly timed; ordinary Julia
garbage collection and finalizers may run inside the construction timer.
These measure construction and ingestion, rather than time to solve the model.

All three conditions use the same frozen compiler-compatible sources: Julia
1.13.3, JuMP `bl/juliac` (version field 1.32.1), MOI `bl/juliac` (1.54.0),
GenOpt `bl/juliac-safe` at `65e7c41` (0.2.5), and HiGHS.jl 1.26.1 / HiGHS 1.15.1.
For the batch conditions, only GenOpt's `_batch_add_constraints` implementation
was replaced with the PR #50 version. The final column also loads the archived
HiGHS row-batch prototype; it does not use the later full `bl/bulk-methods` branch.
No Python frontend or compiled JuMPy library is timed in this comparison.

[Summary CSV](summary.csv) · [All 36 samples](samples.jsonl) ·
[Environment and source hashes](metadata.json) ·
[GenOpt change](GenOpt-bulk.patch) · [HiGHS row-batch prototype](HiGHS-bulk.patch)
