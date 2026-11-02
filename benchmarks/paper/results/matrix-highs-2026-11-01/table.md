Million-variable LQCP; five warmed samples, one thread. All totals include data assembly, row creation where applicable, final loading and the zero-limit HiGHS run.

| Path | Median total | Range |
| --- | ---: | ---: |
| Direct arrays / passModel | 335.9 ms | 335.2–348.3 ms |
| Transpose CSC + MatrixOfConstraints wrapper | 353.4 ms | 335.8–364.7 ms |
| MOI.add_constraints into transposed matrix | 474.3 ms | 470.5–480.3 ms |
| Native addRows, one array batch | 342.2 ms | 326.1–346.7 ms |
| Native addRows, array batches | 356.2 ms | 351.2–358.7 ms |
| Native addRows from the same MOI functions | 446.5 ms | 418.8–455.7 ms |

Independent-process check of the two scalar-function paths:

| Path | Median total |
| --- | ---: |
| MOI.add_constraints into transposed matrix | 534.0 ms |
| Native addRows from the same MOI functions | 489.4 ms |

The matrix path includes its final transfer into HiGHS; comparing append time alone omits that cost. The custom storage handles only equality rows and fixed variable numbering. Objective and variable-bound arrays are supplied by the direct builder.
