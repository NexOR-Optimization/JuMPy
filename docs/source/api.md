# API reference

## Models

Choose `import jumpy.highs as jp` for compiled HiGHS, or
`import jumpy.juliacall as jp` for JuliaCall. Each provides `jp.Model()` with no
backend argument and the same expression helpers. Use one interface per
process; native objects cannot cross between their Julia runtimes.

```{eval-rst}
.. autoclass:: jumpy.highs.Model
   :members:
   :inherited-members:

.. autoclass:: jumpy.juliacall.Model

.. autofunction:: jumpy.minimize

.. autofunction:: jumpy.maximize
```

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

Compiled sets are released automatically. A saved set can be reused across
models of the same interface, even after a model closes. For explicit lifetime
control, call its `close()` method or use it as a context manager; a closed set
cannot be reused.

## Expressions

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
