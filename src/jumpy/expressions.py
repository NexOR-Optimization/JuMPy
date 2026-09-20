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
    """A single decision variable; keeps its column for solution lookup."""

    def __init__(self, ops, index: int, name: str | None = None):
        super().__init__(ops, ops.variable(index))
        self.index = index
        self.name = name

    def __repr__(self) -> str:
        return self.name or f"x[{self.index}]"


class VariableVector:
    """
    A block of decision variables returned by Model.variables().

    Concrete indexing (x[0]) returns a Variable; symbolic indexing (x[i]
    with an expression) builds a getindex template node over the contiguous
    block.
    """

    def __init__(self, ops, start: int, count: int, name: str | None = None):
        self._ops = ops
        self.start = start
        self.count = count
        self.name = name
        self._block = None  # Native JuMP variable array, built lazily.

    def __getitem__(self, index):
        if isinstance(index, int):
            var_name = f"{self.name}[{index}]" if self.name else None
            return Variable(self._ops, self.start + index, var_name)
        if isinstance(index, Node):
            if self._block is None:
                self._block = self._ops.contiguous_variables(self.start, self.count)
            block = Node(self._ops, self._block)
            # 0-based Python index -> 1-based Julia index
            return block._apply("getindex", [block, index + 1])
        raise TypeError(f"Index must be int or Node, got {type(index).__name__}")

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return (self[k] for k in range(self.count))

    def __repr__(self) -> str:
        return f"{self.name or 'x'}[0:{self.count}]"


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


class Objective:
    """An optimization objective (minimize or maximize)."""

    def __init__(self, sense: str, func: Node):
        self.sense = sense
        self.func = func


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
