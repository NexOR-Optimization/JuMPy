LQCP with 1,003,001 variables and 1,002,001 equality rows (n = 1000). Five samples after warming each mode at n = 1000, one thread, presolve off, zero solve time limit.

| Submission | Total | Range |
| --- | ---: | ---: |
| Bulk MOI calls | 986.3 ms | 982.5–1,113.9 ms |
| Scalar MOI calls | 2,626.4 ms | 2,601.0–2,636.9 ms |

Bulk submission is 2.66× faster for this complete model. Both paths use an ordinary HiGHS.Optimizer and the same hardcoded formulation, without JuMP, GenOpt, custom adapters, or raw C API model loading.

Phase medians (milliseconds):

| Submission | Variables | Bounds | Objective | Rows | HiGHS run + results |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bulk | 32.1 | 252.1 | 14.2 | 393.5 | 307.3 |
| Scalar | 472.7 | 715.4 | 17.9 | 1086.4 | 318.3 |

Totals include setup, variable and bound creation, the full quadratic objective, scalar affine function construction, native loading, and MOI.optimize! result storage. Rows use the canonical six-term PDE stencil, in batches of at most 65,536 functions. The scalar path creates the same functions. Compilation, explicit collection before each sample, and disposal are excluded. Phase medians do not necessarily sum to the median total.

HiGHS.jl 1.26.1 with bulk methods from local commit 620f3fc; MOI 1.54.0 from the existing compiler-compatible benchmark environment; Julia 1.13.3. The initial JSON record confirms that all three bulk insertion methods dispatch to HiGHS.jl.
