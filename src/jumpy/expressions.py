"""
Expression nodes, built eagerly as MOI functions.

Every arithmetic operation immediately performs one MOI call through the
model's `ops` object — via juliacall or the compiled library. A Node is a
thin Python handle around the resulting MOI object; there is no Python-side
expression tree and no conversion step.

Iterators (from Model.iterator) are ordinary nodes wrapping a
GenOpt.IteratorRef, so templates like `x[i] + x[i + 1] <= 10` are also
built eagerly; GenOpt discovers the iterators by identity when the group
constraint is added.
"""

from __future__ import annotations

Numeric = (int, float)


def _moi(ops, value):
    """The MOI object of a Node or a numeric literal."""
    if isinstance(value, Node):
        return value.moi
    if isinstance(value, Numeric):
        return ops.constant(float(value))
    raise TypeError(f"Cannot use {type(value).__name__} in an expression")


def _is_linear(value) -> bool:
    return not isinstance(value, Node) or value.linear


class Node:
    """A handle to an MOI expression owned by the model's backend."""

    def __init__(self, ops, moi, *, linear: bool = True):
        self._ops = ops
        self.moi = moi
        self.linear = linear

    def _snf(self, head: str, args, *, linear: bool = True) -> Node:
        return Node(
            self._ops,
            self._ops.scalar_nonlinear(head, [_moi(self._ops, a) for a in args]),
            linear=linear and all(_is_linear(a) for a in args),
        )

    # -- arithmetic (each call is one MOI ScalarNonlinearFunction) --------------

    def __add__(self, other):
        return self._snf("+", [self, other])

    def __radd__(self, other):
        return self._snf("+", [other, self])

    def __sub__(self, other):
        return self._snf("-", [self, other])

    def __rsub__(self, other):
        return self._snf("-", [other, self])

    def __mul__(self, other):
        return self._snf("*", [self, other])

    def __rmul__(self, other):
        return self._snf("*", [other, self])

    def __truediv__(self, other):
        return self._snf("/", [self, other], linear=False)

    def __rtruediv__(self, other):
        return self._snf("/", [other, self], linear=False)

    def __pow__(self, other):
        return self._snf("^", [self, other], linear=False)

    def __rpow__(self, other):
        return self._snf("^", [other, self], linear=False)

    def __neg__(self):
        return self._snf("-", [self])

    def __pos__(self):
        return self

    # -- comparisons (normalized to `self - other  sense  0`) -------------------

    def __le__(self, other) -> Constraint:
        return Constraint(self - other, "<=")

    def __ge__(self, other) -> Constraint:
        return Constraint(self - other, ">=")

    def __eq__(self, other) -> Constraint:
        return Constraint(self - other, "==")


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
        self._block = None  # GenOpt.ContiguousArrayOfVariables, built lazily

    def __getitem__(self, index):
        if isinstance(index, int):
            var_name = f"{self.name}[{index}]" if self.name else None
            return Variable(self._ops, self.start + index, var_name)
        if isinstance(index, Node):
            if self._block is None:
                self._block = self._ops.contiguous_variables(self.start, self.count)
            block = Node(self._ops, self._block)
            # 0-based Python index -> 1-based Julia index
            return block._snf("getindex", [block, index + 1])
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
            return array._snf("getindex", [array, index + 1])
        raise TypeError(f"Index must be int or Node, got {type(index).__name__}")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"{self.name or 'param'}[0:{len(self.values)}]"


class Constraint:
    """A normalized constraint: `func sense 0`."""

    def __init__(self, func: Node, sense: str):
        self.func = func
        self.sense = sense

    def __repr__(self) -> str:
        return f"<constraint: f(x) {self.sense} 0>"


class Objective:
    """An optimization objective (minimize or maximize)."""

    def __init__(self, sense: str, func: Node):
        self.sense = sense
        self.func = func


# -- nonlinear functions --------------------------------------------------------

def _func(name: str, x: Node) -> Node:
    return x._snf(name, [x], linear=False)


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
