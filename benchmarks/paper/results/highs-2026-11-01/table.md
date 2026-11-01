| Model | n | Variables | JuMP cached | JuMP direct | JuMP direct + GenOpt | JuMPy.HiGHS | Python / grouped Julia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LQCP | 100 | 10,301 | 45.9 ms | 36.1 ms | **35.1 ms** | 38.3 ms | 1.09× |
| LQCP | 250 | 63,251 | 326.9 ms | 227.3 ms | **157.5 ms** | 178.8 ms | 1.13× |
| LQCP | 500 | 251,501 | 1,414.4 ms | 877.7 ms | **683.3 ms** | 696.4 ms | 1.02× |
| LQCP | 1000 | 1,003,001 | 6,476.7 ms | 4,083.4 ms | 2,883.7 ms | **2,829.8 ms** | 0.98× |
