# API reference

## Models

Choose `import jumpy.highs as jp` for compiled HiGHS, or
`import jumpy.juliacall as jp` for JuliaCall. Each provides `jp.Model()` with no
backend argument and the same expression helpers. Use one interface per
process; native objects cannot cross between their Julia runtimes.

Compiled HiGHS's `jp.Model()` includes HiGHS. JuliaCall constructors follow
JuMP's solver selection:

- `jp.Model()` calls `JuMP.Model()` and creates a model without a solver.
- `jp.Model(jl.HiGHS.Optimizer)` calls `JuMP.Model(HiGHS.Optimizer)`, using the
  Julia constructor as an optimizer factory.
- `jp.direct_model(jl.HiGHS.Optimizer())` calls
  `JuMP.direct_model(HiGHS.Optimizer())`, using the supplied optimizer instance.

Install and import the selected solver in the JuliaCall environment first.
By default, cached models register GenOpt's bridges. Direct models add neither a cache nor
bridges; their optimizer determines which functions and sets they support.
JuliaCall constructors preserve JuMP's solver-output defaults.

```{eval-rst}
.. autoclass:: jumpy.highs.Model
   :members:
   :inherited-members:

.. autoclass:: jumpy.juliacall.Model

.. autofunction:: jumpy.juliacall.direct_model

.. autofunction:: jumpy.minimize

.. autofunction:: jumpy.maximize
```

## Solve time limits

`model.set_time_limit_sec(seconds)` sets a positive finite limit in seconds for
either backend. `model.unset_time_limit_sec()` clears it before the next solve;
`model.set_time_limit_sec(None)` is equivalent:

```python
model.set_time_limit_sec(60.0)
model.optimize()
model.unset_time_limit_sec()
```

If the limit is reached, `optimize()` returns the termination status normally.
Use `model.has_values()` to check whether a solution is available.

## Function-in-set constraints

Comparison syntax and explicit MOI sets are both supported:

```python
import jumpy.highs as jp

model = jp.Model()
x, y = model.variables(2, lower=0)
model.constraint(x + y, jp.MOI.LessThan(1.0))
model.constraint(x + y >= 0.5)

z = model.variable()
model.constraint(z, jp.MOI.Integer())
```

The explicit function must be a scalar expression, variable, or number.
Comparisons are shorthand: `model.constraint(x <= 1)` delegates to
`model.constraint(x, jp.MOI.LessThan(1.0))`. In either form, a bare variable keeps
MOI's variable-in-set semantics, including bounds and integrality. Use an
expression such as `x + 0` to add an affine row instead of a bound.
Comparisons, variable bounds, and constraint groups all use native MOI sets
internally, just like explicit function-in-set constraints.

`jumpy.highs.MOI` exposes `LessThan(upper)`, `GreaterThan(lower)`,
`EqualTo(value)`, `ZeroOne()`, and `Integer()`. Each creates a native Julia set
behind an opaque Python handle; other constructor names raise a clear error.
`jumpy.juliacall.MOI` is the actual Julia module, with the chosen optimizer and
its bridges determining which constraints it supports.
Use floating-point bounds such as `1.0` for HiGHS sets; comparison operators
convert numeric bounds to floats automatically.

Symbolic comparisons create constraints, not Python booleans. Using them in
`if`, membership tests such as `x in [y]`, or chained comparisons such as
`0 <= x <= 1` raises `TypeError`. Add the two bounds separately, and use
identity checks such as `any(x is variable for variable in variables)` when
checking whether a particular Python variable object is in a collection.

Compiled sets are released automatically. A saved set can be reused across
models of the same interface, even after a model closes. For explicit lifetime
control, call its `close()` method or use it as a context manager; a closed set
cannot be reused.

## Expressions

`model.lazy_sum(template)` creates a native GenOpt `LazySum`, preserving its
expression template and iterators for solvers such as ExaModels. With ExaModels
and MadNLP installed in the JuliaCall environment:

```python
import jumpy.juliacall as jp
from juliacall import Main as jl

jl.seval("using ExaModels, MadNLP")
model = jp.Model(lambda: jl.ExaModels.Optimizer(jl.MadNLP.madnlp))
x = model.variables(3, lower=0)
cost = model.parameter([2, 3, 4])
i = model.iterator(range(3))
model.objective = jp.minimize(model.lazy_sum(cost[i] * x[i]))
```

The zero-argument Python factory lets JuMP construct the optimizer with its
solver function.
The `lazy_sum` argument is one symbolic template, not a Python iterable or
generator. With HiGHS, GenOpt lowers affine lazy sums and quadratic lazy-sum
objectives to ordinary solver functions. Python's built-in `sum()` remains
available for explicit sums.
Direct `model.value(lazy_sum_expression)` queries are not currently supported.

Python operators call JuMP's native arithmetic and keep only opaque references
to the resulting expressions. JuMP determines whether an expression is affine,
quadratic, or nonlinear. HiGHS supports convex quadratic objectives such as
`model.objective = jp.minimize((x - 2) ** 2)`; unsupported functions produce
solver errors.

Integer literals and integer iterator values retain their types. In particular,
`x ** 2` uses JuMP's quadratic arithmetic, while `x ** 2.0` is nonlinear.
The compiled interface requires integer literals and integer-only iterators to fit in
Julia's `Int64`; out-of-range values raise `OverflowError` before the C call.

Variable arrays hold native Julia array references. Concrete indices are local
to their array and follow Python's zero-based and negative-index conventions;
out-of-range indices raise `IndexError`, even if the model has other arrays.

After optimizing, `model.value(x)` or `model.value(2 * x + 1)` queries the
current solution in JuMP directly, without a Python column mapping or solution
cache. Querying variable values before solving raises a backend error.
Expressions must belong to the queried model, and the model must still be open.

## Solve results

`model.optimize()` returns the native MOI termination status code (`1` means
`OPTIMAL`), including nonoptimal termination such as a time limit or
infeasibility. Check solution availability before querying values:

```python
status = model.optimize()
if model.has_values():
    print(model.objective_value())
    print(model.value(x))
```

`model.termination_status()` queries the current status without solving.
`model.has_values()` and `model.has_duals()` query primal and dual result
availability; MIPs normally have no dual solution. `model.objective_value()`
reads the solver's objective value rather than reconstructing it in Python.
These queries follow JuMP's result semantics: modifying the JuliaCall backend's
cached model invalidates its results, while the direct HiGHS backend leaves
result availability to the solver. Reoptimize after modifying the model before
using its solution; `model.value(...)` rejects queries until then. Querying an
unavailable objective value raises a backend error.

## Constraint groups

Symbolic indexing creates native GenOpt expression templates.
`model.constraint_group(con)` delegates to `model.constraint(con)`, which uses
JuMP's constraint construction and GenOpt's bridge to expand iterator-based
constraints. There is no separate Python group insertion or classification path.

```{eval-rst}
.. autoclass:: jumpy.Variable
   :members:

.. autoclass:: jumpy.VariableVector
   :members:

.. autoclass:: jumpy.Parameter
   :members:

.. autoclass:: jumpy.Node
   :members:

.. autoclass:: jumpy.Constraint
   :members:

.. autoclass:: jumpy.Objective
   :members:
```

## Nonlinear functions

```{eval-rst}
.. autofunction:: jumpy.sin

.. autofunction:: jumpy.cos

.. autofunction:: jumpy.exp

.. autofunction:: jumpy.log

.. autofunction:: jumpy.sqrt

.. autofunction:: jumpy.jp_abs
```
