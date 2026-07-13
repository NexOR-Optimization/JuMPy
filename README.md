# JuMPy

A Python interface to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) via [GenOpt](https://github.com/blegat/GenOpt.jl).

JuMPy lets you build optimization models in Python at the speed of compiled Julia. It does this by constructing lightweight expression templates in Python and handing them off to a compiled Julia backend for constraint expansion and solving — keeping the expensive work out of Python entirely.

## Why JuMPy?

Python modeling libraries like Pyomo and CVXPY construct models at Python speed. For large-scale problems with millions of constraints, model construction can take longer than solving. JuMPy eliminates this bottleneck.

The key idea: most large models are a small number of **constraint groups** — a parametric template repeated over a large index set. JuMPy builds one expression template per group in Python, then [GenOpt](https://github.com/blegat/GenOpt.jl) expands it into individual constraints in compiled Julia.

```mermaid
flowchart LR
    subgraph Python ["Python (JuMPy)"]
        templates["Few expression templates\n(built once, cheap)"]
        solution["Solution values"]
    end

    subgraph Julia ["Compiled Julia (juliac)"]
        genopt["GenOpt\nexpands into millions\nof constraints (fast)"]
        moi["MathOptInterface"]
        highs["HiGHS solver"]
        result["Result vector"]

        genopt --> moi --> highs --> result
    end

    templates -- "FFI" --> genopt
    result -- "FFI" --> solution
```

The Python workload is proportional to the **number of groups**, not the number of constraints.

## Installation

```
pip install jumpy
```

No Julia installation required — JuMPy ships a precompiled solver backend built with [juliac](https://docs.julialang.org/en/v1/devdocs/juliac/).

## Quick start

```python
import jumpy as jp

m = jp.Model()
x = m.variables(100, lower=0, name="x")

# A constraint group: 99 constraints from one template
i = m.iterator(range(99))
m.constraint_group(x[i] + x[i + 1] <= 10)

m.objective = jp.minimize(x[0] + x[1])
m.optimize()

print(m.value(x[0]))
```

## Constraint groups

Constraint groups are the core feature. Instead of building constraints one by one in Python, you write a single expression template with symbolic iterators.

### Basic group

```python
i = m.iterator(range(1000000))
m.constraint_group(x[i] <= 10)
# One template in Python → 1,000,000 constraints in Julia
```

### Multi-dimensional

```python
i = m.iterator(range(100))
j = m.iterator(range(100))
m.constraint_group(x[100 * i + j] >= 0)
# 10,000 constraints from one template
```

### With data

```python
costs = m.parameter([...], name="costs")
demand = m.parameter([...], name="demand")

i = m.iterator(range(n))
m.constraint_group(costs[i] * x[i] >= demand[i])
```

### Nonlinear

```python
i = m.iterator(range(n))
m.constraint_group(jp.sin(x[i]) + jp.exp(x[i]) <= 1.0)
```

### Individual constraints

For one-off constraints that don't need grouping:

```python
m.constraint(x[0] + x[1] == 5)
```

## API reference

### Model

| Method | Description |
|---|---|
| `m = jp.Model()` | Create a new model |
| `m.variables(n, lower=, upper=, name=, binary=, integer=)` | Add `n` variables, returns a `VariableVector` |
| `m.variable(lower=, upper=, name=, binary=, integer=)` | Add a single variable |
| `m.constraint_group(template)` | Add a constraint group (iterators are discovered from the template) |
| `m.constraint(con)` | Add an individual constraint |
| `m.objective = jp.minimize(expr)` | Set a minimization objective |
| `m.objective = jp.maximize(expr)` | Set a maximization objective |
| `m.iterator(range(n))` | An index set for constraint groups |
| `m.parameter(values, name=)` | A data vector, symbolically indexable |
| `m.optimize()` | Solve the model |
| `m.value(var)` | Get the solved value of a variable |

### Expressions

Variables and iterators support standard arithmetic (`+`, `-`, `*`, `/`, `**`) and comparisons (`<=`, `>=`, `==`). Nonlinear functions are available as:

```python
jp.sin(x)   jp.cos(x)   jp.exp(x)
jp.log(x)   jp.sqrt(x)  jp.jp_abs(x)
```

### Symbolic indexing

`VariableVector` and `Parameter` support both concrete and symbolic indexing:

```python
x[0]          # concrete: returns a Variable
x[i]          # symbolic: a getindex template node over the block
x[10*i + j]   # symbolic arithmetic on the index

costs[0]      # concrete: returns a float
costs[i]      # symbolic: a getindex template node over the data
```

## Architecture

JuMPy has two layers:

1. **Python package** (`jumpy`): Operator overloading builds MOI functions *eagerly* — every operation is one MOI call through the backend's `ops` object. There is no Python-side expression tree and no conversion step.

2. **Compiled Julia library** (built with juliac): exposes the MOI API as C entry points, one per MOI call — `jumpy_scalar_nonlinear` is the compiled `MOI.ScalarNonlinearFunction` constructor, and so on. GenOpt is compiled in: templates reference iterators by identity (`GenOpt.IteratorRef`) and groups are expanded in Julia.

```
src/jumpy/
├── expressions.py        # Node handles with operator overloading (eager MOI calls)
├── bridge_juliacall.py   # MOI ops via juliacall
├── backend.py            # Backend selection; MOI ops via ctypes (juliac)
└── model.py              # Model class: variables, groups, objective, solve
```

## How it maps to Julia

| Python | Julia |
|---|---|
| `m.iterator(range(n))` | `GenOpt.IteratorRef(GenOpt.Iterator(values))` |
| `x[i]` (symbolic) | `getindex` node over `GenOpt.ContiguousArrayOfVariables` |
| `x[i] + x[i+1] <= 10` | `MOI.ScalarNonlinearFunction` template |
| `m.constraint_group(...)` | `GenOpt.FunctionGenerator` (iterators discovered by identity) |
| `m.parameter([...])` | Data vector, `getindex` resolved during expansion |

## Development

```bash
# Run tests
python3 tests/test_expressions.py

# Run example
python3 examples/basic.py

# Build the compiled backend (see julia/README.md), then:
JUMPY_BACKEND=juliac python3 tests/test_solve.py
```

The compiled backend lives in [`julia/`](julia/): a small Julia package
(`JuMPyHiGHS`) exposing C entry points that mirror the MOI API around a raw
`HiGHS.Optimizer`, compiled with
[JuliaC](https://github.com/JuliaLang/JuliaC.jl). See
[`julia/README.md`](julia/README.md) for the C ABI and build instructions.

## Related projects

- [JuMP](https://github.com/jump-dev/JuMP.jl) — the Julia optimization modeling language
- [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) — JuMP's solver abstraction layer
- [GenOpt](https://github.com/blegat/GenOpt.jl) — constraint group expansion
- [HiGHS](https://highs.dev/) — open-source LP/MIP solver

## License

MIT
