Million-variable LQCP (n = 1000, 1,003,001 variables, 1,002,001 equality rows). Medians of five samples after warming each path at n = 1000; one thread, presolve off, zero solve time limit.

For a comparison isolating GenOpt PR #50, see the [batch-submission table](../genopt-bulk-constraints-2026-11-01/table.md).

| Path | Total | Range |
| --- | ---: | ---: |
| Hardcoded arrays → native addRows | 335.9 ms | 326.2–341.7 ms |
| Hardcoded scalar functions → native addRows | 450.9 ms | 429.7–455.4 ms |
| GenOpt → scalar functions → native addRows | 664.4 ms | 635.3–692.9 ms |
| GenOpt → native row buffers | 503.5 ms | 501.6–527.3 ms |
| GenOpt → scalar functions + HiGHS.jl mapping/metadata | 1,265.2 ms | 1,239.2–1,274.6 ms |
| Complete grouped JuMP model, installed packages | 2,632.2 ms | 2,620.0–2,644.2 ms |
| Complete grouped model, native row batches | 2,049.7 ms | 2,026.9–2,071.8 ms |
| Complete grouped model, direct row buffers | 1,919.4 ms | 1,913.2–1,977.2 ms |
| Direct row buffers + bulk bounds | 1,448.0 ms | 1,413.7–1,463.1 ms |
| Direct row buffers + bulk bounds and variables | 953.9 ms | 952.3–1,014.6 ms |
| Linopy | 980.1 ms | 932.6–1,006.1 ms |

The first five paths retain hardcoded objective and variable-bound arrays; their setup also assembles the reference row arrays. The complete grouped paths build all variables, bounds, objectives and constraints through the same JuMP/GenOpt formulation. Their totals include HiGHS.jl result storage through MOI.optimize!. Raw paths call Highs_run directly. No MatrixOfConstraints is used.

Phase medians for the complete experimental paths (milliseconds):

| Path | Variables | Bounds | Scalar functions | Row packing | Native addRows | Row metadata | HiGHS run + results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete grouped model, native row batches | 477.8 | 711.3 | 84.0 | 170.9 | 73.6 | 19.5 | 227.7 |
| Complete grouped model, direct row buffers | 481.9 | 707.9 | 0.0 | 88.6 | 73.0 | 21.9 | 230.8 |
| Direct row buffers + bulk bounds | 479.9 | 216.6 | 0.0 | 89.0 | 73.3 | 21.8 | 227.6 |
| Direct row buffers + bulk bounds and variables | 35.0 | 216.8 | 0.0 | 197.0 | 73.0 | 21.1 | 230.9 |

Objective updates, template planning, JuMP reference creation/checks, and remaining Julia overhead are included in totals. Phase medians do not necessarily sum to the median total; allocation-triggered garbage collection is charged to the phase where it occurs.

Compilation, explicit collection before each Julia sample, and model disposal are excluded. Linopy was refreshed in a separate process with the same native HiGHS version and settings; it uses its ordinary Python garbage collection. These are experimental adapters using GenOpt and HiGHS.jl internals, not changes to the released JuMPy backend.
