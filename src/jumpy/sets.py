"""Scalar constraint sets supported by JuMPy's backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LessThan:
    """The scalar set ``{x : x <= upper}``."""

    upper: float


@dataclass(frozen=True)
class GreaterThan:
    """The scalar set ``{x : x >= lower}``."""

    lower: float


@dataclass(frozen=True)
class EqualTo:
    """The scalar set ``{x : x == value}``."""

    value: float


@dataclass(frozen=True)
class ZeroOne:
    """The binary set ``{0, 1}``, applied to a single variable."""


@dataclass(frozen=True)
class Integer:
    """The set of integers, applied to a single variable."""


ScalarSet = LessThan | GreaterThan | EqualTo | ZeroOne | Integer


def _constraint_args(set_: ScalarSet) -> tuple[str, float]:
    """Encode an existing scalar set for the backend's sense/rhs interface."""
    # Exact types: a custom set (including a subclass) must not silently
    # acquire the semantics of one of the sets compiled into the C ABI.
    if type(set_) is LessThan:
        return "<=", float(set_.upper)
    if type(set_) is GreaterThan:
        return ">=", float(set_.lower)
    if type(set_) is EqualTo:
        return "==", float(set_.value)
    if type(set_) is ZeroOne:
        return "binary", 0.0
    if type(set_) is Integer:
        return "integer", 0.0
    raise TypeError(
        f"Unsupported constraint set: {type(set_).__name__}. "
        "Expected LessThan, GreaterThan, EqualTo, ZeroOne, or Integer."
    )
