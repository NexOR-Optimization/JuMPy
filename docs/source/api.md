# API reference

## Models

```{eval-rst}
.. autoclass:: jumpy.Model
   :members:

.. autofunction:: jumpy.minimize

.. autofunction:: jumpy.maximize
```

## Constraint sets

Use either comparison syntax or an explicit scalar set:

```python
m.constraint(x + y <= 1)
# Equivalently:
m.constraint(x + y, jp.LessThan(1))
```

Passing a variable directly, for example `m.constraint(x, jp.Integer())`,
adds a variable bound or domain. Vector and custom sets are not yet supported.

```{eval-rst}
.. autoclass:: jumpy.LessThan

.. autoclass:: jumpy.GreaterThan

.. autoclass:: jumpy.EqualTo

.. autoclass:: jumpy.ZeroOne

.. autoclass:: jumpy.Integer
```

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
