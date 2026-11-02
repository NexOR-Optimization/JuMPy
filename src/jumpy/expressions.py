"""
Opaque handles to native JuMP and GenOpt expressions.

Every arithmetic operation immediately calls a Julia operator through the
model's `ops` object — via juliacall or the compiled library. A Node is a
thin Python handle around the resulting Julia object; expression types and
coefficient promotion are owned by JuMP, not classified in Python.

Iterators and indexed expressions use GenOpt's native operator overloads.
"""

from __future__ import annotations

Numeric = (int, float)


def _native(ops, value):
    """The native Julia object of a Node or a numeric literal."""
    if isinstance(value, Node):
        if value._ops is not ops:
            raise ValueError("Cannot combine expressions from different models")
        return value.ref
    if isinstance(value, Numeric):
        return ops.constant(value)
    raise TypeError(f"Cannot use {type(value).__name__} in an expression")


class Node:
    """A handle to a Julia expression owned by the model's backend."""

    def __init__(self, ops, ref):
        self._ops = ops
        self.ref = ref

    def _apply(self, op: str, args) -> Node:
        return Node(
            self._ops,
            self._ops.apply(op, [_native(self._ops, a) for a in args]),
        )

    # -- arithmetic (Julia decides the resulting expression type) -------------

    def __add__(self, other):
        return self._apply("+", [self, other])

    def __radd__(self, other):
        return self._apply("+", [other, self])

    def __sub__(self, other):
        return self._apply("-", [self, other])

    def __rsub__(self, other):
        return self._apply("-", [other, self])

    def __mul__(self, other):
        return self._apply("*", [self, other])

    def __rmul__(self, other):
        return self._apply("*", [other, self])

    def __truediv__(self, other):
        return self._apply("/", [self, other])

    def __rtruediv__(self, other):
        return self._apply("/", [other, self])

    def __pow__(self, other):
        return self._apply("^", [self, other])

    def __rpow__(self, other):
        return self._apply("^", [other, self])

    def __neg__(self):
        return self._apply("-", [self])

    def __pos__(self):
        return self

    # -- comparisons: shorthand for a function in a native set -----------------

    def _comparison(self, other, set_type) -> Constraint:
        if isinstance(other, Numeric):
            return Constraint(self, set_type(float(other)))
        return Constraint(self - other, set_type(0.0))

    def __le__(self, other) -> Constraint:
        return self._comparison(other, self._ops.MOI.LessThan)

    def __ge__(self, other) -> Constraint:
        return self._comparison(other, self._ops.MOI.GreaterThan)

    def __eq__(self, other) -> Constraint:
        return self._comparison(other, self._ops.MOI.EqualTo)


class Variable(Node):
    """An opaque handle to a native JuMP variable."""

    def __init__(self, ops, ref, name: str | None = None):
        super().__init__(ops, ref)
        self.name = name

    def __repr__(self) -> str:
        return self.name or "Variable()"


class VariableArray(Node):
    """
    A native Julia array of variables returned by Model.variables().

    Concrete indexing (x[0], or x[0, 1] for a matrix) returns a Variable, and
    slices (x[:, 1]) return a VariableArray of the same variables. Symbolic
    indexing of a vector (x[i] with an expression) builds a GenOpt template
    over the same array.
    """

    def __init__(self, ops, ref, shape, name: str | None = None):
        super().__init__(ops, ref)
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.name = name

    @property
    def count(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        indices = index if isinstance(index, tuple) else (index,)
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        if len(indices) == 1 and isinstance(index, Node):
            # 0-based Python array index -> 1-based Julia array index.
            return self._apply("getindex", [self, index + 1])
        bounds = []
        for i, n in zip(indices, self.shape):
            if isinstance(i, slice):
                start, stop, step = i.indices(n)
                if step != 1:
                    raise IndexError("Slices of variable arrays have a step of 1")
                bounds.append((start + 1, max(start, stop)))
            elif isinstance(i, int):
                if i < 0:
                    i += n
                if not 0 <= i < n:
                    raise IndexError("Variable array index out of range")
                bounds.append((i + 1, None))
            else:
                raise TypeError(
                    f"Index must be int, slice or Node, got {type(i).__name__}"
                )
        label = f"{self.name or 'x'}[{', '.join(map(str, indices))}]"
        if all(last is None for _, last in bounds):
            if len(bounds) == 1:
                ref = self._apply("getindex", [self, bounds[0][0]]).ref
            else:
                ref = self._ops.getindex(self.ref, bounds)
            return Variable(self._ops, ref, label)
        shape = [last - first + 1 for first, last in bounds if last is not None]
        return VariableArray(self._ops, self._ops.getindex(self.ref, bounds), shape, label)

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        if len(self.shape) != 1:
            raise TypeError("Only vectors of variables are iterable; index matrices")
        return (self[k] for k in range(self.count))

    def __repr__(self) -> str:
        return f"{self.name or 'x'}[{', '.join(f'0:{n}' for n in self.shape)}]"


# Variable vectors are one-dimensional variable arrays.
VariableVector = VariableArray


class Parameter:
    """
    A vector of constant data returned by Model.parameter().

    Concrete indexing (costs[0]) returns a float; symbolic indexing
    (costs[i]) builds a getindex template node over the data vector.
    """

    def __init__(self, ops, values, name: str | None = None):
        self._ops = ops
        self.values = [float(v) for v in values]
        self.name = name
        self._array = None  # data vector node, built lazily

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]
        if isinstance(index, Node):
            if self._array is None:
                self._array = self._ops.float_array(self.values)
            array = Node(self._ops, self._array)
            return array._apply("getindex", [array, index + 1])
        raise TypeError(f"Index must be int or Node, got {type(index).__name__}")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"{self.name or 'param'}[0:{len(self.values)}]"


class Constraint:
    """A scalar expression in a native MOI set."""

    def __init__(self, func: Node, set_):
        self.func = func
        self.set = set_

    def __repr__(self) -> str:
        return f"<constraint: f(x) in {self.set}>"

    def __bool__(self):
        raise TypeError(
            "A symbolic constraint has no truth value. Add it with model.constraint(); "
            "do not use symbolic comparisons in Python conditions or membership tests."
        )


class ConstraintRef:
    """An opaque reference to an added scalar constraint or constraint group."""

    def __init__(self, ops, ref):
        self._ops = ops
        self.ref = ref


class Objective:
    """An optimization objective (minimize or maximize)."""

    def __init__(self, sense: str, func: Node):
        self.sense = sense
        self.func = func


def vcat(*args) -> Node:
    """
    The vector of affine expressions concatenating ``args``, as ``[a; b; c]``
    in Julia: numbers, scalar expressions, vectors of variables or affine
    expressions, and Python sequences of these.

    Example:
        model.constraint(jp.vcat(t, depot, nodes[:, 0], depot), set_)
    """
    items = list(_flatten(args))
    ops = next((item._ops for item in items if isinstance(item, Node)), None)
    if ops is None:
        raise TypeError("vcat needs at least one JuMPy expression")
    return Node(ops, ops.vcat([_native(ops, item) for item in items]))


def _flatten(args):
    for arg in args:
        if isinstance(arg, (list, tuple)):
            yield from _flatten(arg)
        else:
            yield arg


# -- nonlinear functions --------------------------------------------------------

def _func(name: str, x: Node) -> Node:
    return x._apply(name, [x])


def sin(x):
    return _func("sin", x)

def cos(x):
    return _func("cos", x)

def exp(x):
    return _func("exp", x)

def log(x):
    return _func("log", x)

def sqrt(x):
    return _func("sqrt", x)

def abs(x):
    return _func("abs", x)
