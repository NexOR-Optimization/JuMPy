"""Thin bindings to the native sets and operators of the compiled backends.

Each compiled library describes the sets that it constructs in its registry:
their Julia module and name, such as ``MathOptInterface.LessThan``, and the
types of their fields; and its nonlinear operators, such as
``MathOptVRP.op_sum_distances``. ``jp.MOI`` and the other module namespaces of
a facade call these constructors. Only ownership lives in Python; set
definitions and construction stay in Julia.
"""

import ctypes
from collections.abc import Sequence

from jumpy import backend


class NativeSet:
    """An owned native set; automatically released, or closed explicitly."""

    def __init__(self, lib, handle):
        if not handle:
            raise RuntimeError("Failed to construct the native MOI set")
        self._lib = lib
        self._handle = int(handle)

    @property
    def handle(self):
        if not self._handle:
            raise RuntimeError("The native MOI set has been closed")
        return self._handle

    def close(self):
        handle = getattr(self, "_handle", 0)
        if handle:
            if self._lib.jumpy_free_set(handle) != 0:
                raise RuntimeError("Failed to release the native MOI set")
            self._handle = 0

    def __enter__(self):
        self.handle
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# The ctypes element of each scalar type of the registry. Booleans are bytes.
_ELEMENTS = {"Int64": ctypes.c_longlong, "Float64": ctypes.c_double, "Bool": ctypes.c_uint8}
_CONVERSIONS = {"Int64": int, "Float64": float, "Bool": bool}


def _rows(value, name):
    if hasattr(value, "tolist"):  # NumPy arrays
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Expected a sequence for {name}, got {type(value).__name__}")
    return list(value)


def _argument(code, value, name):
    """The ctypes data of a field value, and its numbers of rows and columns."""
    if code in _ELEMENTS:
        return (_ELEMENTS[code] * 1)(_CONVERSIONS[code](value)), (1, 1)
    container, element = code[:-1].split("{")
    convert = _CONVERSIONS[element]
    rows = _rows(value, name)
    if container == "Vector":
        data = [convert(x) for x in rows]
        shape = (len(data), 1)
    else:
        # Matrices are given by rows, as nested lists or NumPy arrays, and
        # stored column-major, as in Julia.
        rows = [_rows(row, name) for row in rows]
        columns = len(rows[0]) if rows else 0
        if any(len(row) != columns for row in rows):
            raise ValueError(f"The rows of {name} must have the same length")
        data = [convert(row[j]) for j in range(columns) for row in rows]
        shape = (len(rows), columns)
    return (_ELEMENTS[element] * len(data))(*data), shape


class SetConstructor:
    """Construct one set type of a compiled backend from its fields."""

    def __init__(self, backend_name, index, name, fields, types):
        self._backend = backend_name
        self._index = index
        self.name = name
        self.fields = fields
        self._types = types

    def __repr__(self):
        return f"{self.name}({', '.join(self.fields)})"

    def __call__(self, *args, **kwargs):
        if len(args) > len(self.fields):
            raise TypeError(f"{self!r} takes {len(self.fields)} arguments")
        values = dict(zip(self.fields, args))
        for key, value in kwargs.items():
            if key not in self.fields or key in values:
                raise TypeError(f"Unexpected or repeated argument {key!r} for {self!r}")
            values[key] = value
        missing = [field for field in self.fields if field not in values]
        if missing:
            raise TypeError(f"{self!r} is missing {', '.join(missing)}")
        data = [
            _argument(code, values[field], field)
            for field, code in zip(self.fields, self._types)
        ]
        pointers = (ctypes.c_void_p * len(data))(
            *(ctypes.cast(array, ctypes.c_void_p) for array, _ in data)
        )
        sizes = (ctypes.c_longlong * (2 * len(data)))(
            *(n for _, shape in data for n in shape)
        )
        lib = backend._load_lib(self._backend)
        return NativeSet(lib, lib.jumpy_new_set(self._index, pointers, sizes, len(data)))


class SetFamily:
    """The set types of the same name with different value parameters, such as
    ``TimeWindows{WITHOUT_START_TIME}``. Select one with
    ``family["WITHOUT_START_TIME"]``; a family with a single member is also
    called directly."""

    def __init__(self, name):
        self.name = name
        self._members = {}

    def __repr__(self):
        return f"<sets {self.name}: {', '.join(map(repr, self._members.values()))}>"

    def __getitem__(self, parameters):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        key = ",".join(str(p) for p in parameters)
        if key not in self._members:
            raise KeyError(f"{self.name}{{{key}}} is not available: {list(self._members)}")
        return self._members[key]

    def __call__(self, *args, **kwargs):
        if len(self._members) != 1:
            raise TypeError(f"Select the parameters of {self.name} with {self.name}[...]")
        return next(iter(self._members.values()))(*args, **kwargs)


class NonlinearOperator:
    """A ``JuMP.NonlinearOperator`` of a compiled backend, such as
    ``jp.MathOptVRP.op_sum_distances(distances, jp.vcat(depot, route, depot))``.

    Its arguments are expressions, numbers, and sequences of numbers, which are
    passed as data vectors or matrices (given by rows).
    """

    def __init__(self, name, head):
        self.name = name
        self.head = head

    def __repr__(self):
        return f"<operator {self.name}>"

    def __call__(self, *args):
        from jumpy.expressions import Node, _native

        nodes = [arg for arg in args if isinstance(arg, Node)]
        if not nodes:
            raise TypeError(f"{self.name} needs at least one JuMPy expression argument")
        ops = nodes[0]._ops
        refs = [
            ops.array(arg) if _is_sequence(arg) else _native(ops, arg) for arg in args
        ]
        return Node(ops, ops.nonlinear(self.head, refs))


def _is_sequence(value):
    return hasattr(value, "tolist") or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    )


# The parsed registry of each backend: module name -> name -> set family or
# nonlinear operator.
_REGISTRIES = {}


def registry(backend_name):
    """The sets of the compiled backend jumpy.<backend_name> by Julia module."""
    if backend_name not in _REGISTRIES:
        lib = backend._load_lib(backend_name)
        modules = {}
        for line in lib.jumpy_registry().decode().splitlines():
            kind, index, name, fields, types = line.split("\t")
            module, _, full = name.partition(".")
            if kind == "operator":
                modules.setdefault(module, {})[full] = NonlinearOperator(name, fields)
                continue
            if kind != "set":
                continue
            base, _, parameters = full.partition("{")
            sets = modules.setdefault(module, {})
            family = sets.setdefault(base, SetFamily(f"{module}.{base}"))
            family._members[parameters.rstrip("}")] = SetConstructor(
                backend_name,
                int(index),
                name,
                fields.split(",") if fields else [],
                types.split(";") if types else [],
            )
        _REGISTRIES[backend_name] = modules
    return _REGISTRIES[backend_name]


class Namespace:
    """The sets and nonlinear operators of the Julia module `module` in the
    compiled backend jumpy.<backend_name>, such as ``jp.MOI.LessThan(1.0)``.

    The library is only loaded when a member is first accessed.
    """

    def __init__(self, backend_name, module):
        self._backend = backend_name
        self._module = module

    def _members(self):
        return registry(self._backend).get(self._module, {})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        members = self._members()
        if name not in members:
            raise AttributeError(
                f"{self._module}.{name} is not exposed by the compiled "
                f"jumpy.{self._backend} backend. Use jumpy.juliacall for other constructors."
            )
        return members[name]

    def __dir__(self):
        return sorted(self._members())

    def __repr__(self):
        return f"<{self._module} sets of jumpy.{self._backend}>"
