| Model | n | Variables | JuMP cached | JuMP direct | JuMP direct + GenOpt | JuMPy.HiGHS | highspy | Pyomo | Linopy | JuMPy / grouped Julia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LQCP | 100 | 10,301 | 59.8 ms | 48.4 ms | 40.3 ms | 38.8 ms | **19.7 ms** | 649.8 ms | 171.7 ms | 0.96× |
| LQCP | 250 | 63,251 | 366.2 ms | 229.3 ms | 170.0 ms | 175.5 ms | **62.6 ms** | 4,614.0 ms | 206.3 ms | 1.03× |
| LQCP | 500 | 251,501 | 1,596.7 ms | 1,074.5 ms | 777.2 ms | 718.9 ms | **251.7 ms** | 24,438.5 ms | 347.6 ms | 0.92× |
| LQCP | 1000 | 1,003,001 | 6,870.7 ms | 4,397.4 ms | 2,982.4 ms | 2,873.8 ms | 1,120.8 ms | 186,056.7 ms | **1,078.3 ms** | 0.96× |
