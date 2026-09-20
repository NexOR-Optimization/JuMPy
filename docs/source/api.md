# API reference

## Models

```{eval-rst}
.. autoclass:: jumpy.Model
   :members:

.. autofunction:: jumpy.minimize

.. autofunction:: jumpy.maximize
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

## Native MOI constructors

`MOIConstructors` creates actual Julia sets and function nodes in the compiled
image. Python owns checked handles, not duplicate definitions of those types.
Use context managers (or `close()`) to release handles:

```python
from jumpy.moi import MOIConstructors

native = MOIConstructors()
with native.scalar_set("<=", 1.0) as owned:
    set_ = owned.to_julia()
# JuliaCall retains the actual set after the handle is released.
assert set_.upper == 1.0
```

`to_julia()` requires JuliaCall using the same untrimmed image/runtime.
Construction without `to_julia()` needs no JuliaCall dependency. Handles are
image-local and invalid after release. Variable indices in this low-level API
are MOI's one-based indices. The public `Model.constraint(func, set)` overload
is intentionally deferred.

```{eval-rst}
.. autoclass:: jumpy.moi.MOIConstructors
   :members:

.. autoclass:: jumpy.moi.MOIObject
   :members:
```
