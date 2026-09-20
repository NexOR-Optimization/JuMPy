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

The explicit function must be a scalar expression, variable, or number. A bare
variable keeps MOI's variable-in-set semantics, including bounds and
integrality. Comparison constraints retain their affine-row semantics.

`jumpy.highs.MOI` exposes `LessThan(upper)`, `GreaterThan(lower)`,
`EqualTo(value)`, `ZeroOne()`, and `Integer()`. Each creates a native Julia set
behind an opaque Python handle; other constructor names raise a clear error.
`jumpy.juliacall.MOI` is the actual Julia module, with the chosen optimizer and
its bridges determining which constraints it supports.

Compiled sets are released automatically. A saved set can be reused across
models of the same interface, even after a model closes. For explicit lifetime
control, call its `close()` method or use it as a context manager; a closed set
cannot be reused.

## Expressions

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
