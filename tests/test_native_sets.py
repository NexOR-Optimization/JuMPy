"""Native set handles and explicit constraint forwarding, without Julia."""

import gc
import weakref

import pytest

import jumpy.highs as jp
from jumpy import backend
from jumpy._highs_moi import NativeSet
from jumpy.model import Model
from mock_ops import MockOps


class FakeLib:
    def __init__(self):
        self.calls = []
        self.next_handle = 2**40
        self.fail_construction = False
        self.free_status = 0
        self.constraint_status = 0
        self.expression_ref = 456
        self.value_result = 1.25

    def _set(self, name, *args):
        self.calls.append((name, *args))
        self.next_handle += 1
        return 0 if self.fail_construction else self.next_handle

    def jumpy_less_than(self, rhs):
        return self._set("LessThan", rhs)

    def jumpy_greater_than(self, rhs):
        return self._set("GreaterThan", rhs)

    def jumpy_equal_to(self, rhs):
        return self._set("EqualTo", rhs)

    def jumpy_zero_one(self):
        return self._set("ZeroOne")

    def jumpy_integer(self):
        return self._set("Integer")

    def jumpy_free_set(self, handle):
        self.calls.append(("free_set", handle))
        return self.free_status

    def jumpy_new_model(self):
        return 99

    def jumpy_free_model(self, model):
        return 0

    def jumpy_add_constraint(self, model, func, set_handle):
        self.calls.append(("constraint", model, func, set_handle))
        return self.constraint_status

    def jumpy_apply(self, model, op, args, nargs):
        self.calls.append(("apply", model, op, tuple(args[:nargs])))
        return self.expression_ref

    def jumpy_integer_constant(self, model, value):
        self.calls.append(("integer_constant", model, value))
        return self.expression_ref

    def jumpy_constant(self, model, value):
        self.calls.append(("constant", model, value))
        return self.expression_ref

    def jumpy_integer_iterator(self, model, values, count):
        self.calls.append(("integer_iterator", model, tuple(values[:count])))
        return self.expression_ref

    def jumpy_iterator(self, model, values, count):
        self.calls.append(("iterator", model, tuple(values[:count])))
        return self.expression_ref

    def jumpy_variables(self, model, count):
        self.calls.append(("variables", model, count))
        return self.expression_ref

    def jumpy_value(self, model, expr):
        self.calls.append(("value", model, expr))
        return self.value_result


@pytest.fixture
def lib(monkeypatch):
    library = FakeLib()
    monkeypatch.setattr(backend, "_load_lib", lambda: library)
    return library


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("LessThan", (1.5,)),
        ("GreaterThan", (2.5,)),
        ("EqualTo", (3.5,)),
        ("ZeroOne", ()),
        ("Integer", ()),
    ],
)
def test_scalar_sets_call_named_julia_constructors(lib, name, args):
    with getattr(jp.MOI, name)(*args) as set_:
        assert isinstance(set_, NativeSet)
        assert set_.handle > 2**32
        assert lib.calls == [(name, *args)]
        # Python owns a handle, not a duplicated definition of the MOI set.
        for field in ("upper", "lower", "value"):
            assert not hasattr(set_, field)


def test_unsupported_constructor_fails_without_loading_library(monkeypatch):
    def unexpected_load():
        pytest.fail("Unsupported constructors must fail before loading Julia")

    monkeypatch.setattr(backend, "_load_lib", unexpected_load)
    with pytest.raises(AttributeError, match="compiled HiGHS backend"):
        jp.MOI.PositiveSemidefiniteConeTriangle(2)


def test_native_set_close_is_idempotent(lib):
    set_ = jp.MOI.LessThan(1.0)
    handle = set_.handle
    set_.close()
    set_.close()
    assert lib.calls == [("LessThan", 1.0), ("free_set", handle)]
    with pytest.raises(RuntimeError, match="closed"):
        _ = set_.handle
    with pytest.raises(RuntimeError, match="closed"):
        with set_:
            pytest.fail("A closed set must not enter its context")


def test_native_set_context_releases_when_body_raises(lib):
    set_ = jp.MOI.LessThan(1.0)
    handle = set_.handle
    with pytest.raises(ValueError, match="example"):
        with set_ as entered:
            assert entered is set_
            raise ValueError("example")
    assert lib.calls[-1] == ("free_set", handle)


def test_native_set_finalizer_releases_handle(lib):
    set_ = jp.MOI.LessThan(1.0)
    handle = set_.handle
    reference = weakref.ref(set_)
    del set_
    gc.collect()
    assert reference() is None
    assert lib.calls[-1] == ("free_set", handle)


def test_null_set_handle_is_a_clean_constructor_error(lib):
    lib.fail_construction = True
    with pytest.raises(RuntimeError, match="construct"):
        jp.MOI.LessThan(1.0)
    assert lib.calls == [("LessThan", 1.0)]


def test_set_release_error_is_reported(lib):
    set_ = jp.MOI.LessThan(1.0)
    lib.free_status = -1
    with pytest.raises(RuntimeError, match="release"):
        set_.close()
    lib.free_status = 0
    set_.close()


def test_compiled_ops_pass_set_handle_without_copying(lib):
    ops = backend.JuliacOps(lib)
    with jp.MOI.LessThan(1.0) as set_:
        ops.add_constraint(123, set_)
        assert lib.calls[-1] == ("constraint", 99, 123, set_.handle)
    ops.free()


@pytest.mark.parametrize(
    ("op", "args"), [("+", [123, 234]), ("^", [123, 234]), ("sin", [123])],
)
def test_compiled_ops_forward_native_operations(lib, op, args):
    ops = backend.JuliacOps(lib)
    assert ops.apply(op, args) == lib.expression_ref
    assert lib.calls == [("apply", 99, op.encode(), tuple(args))]
    ops.free()


def test_compiled_ops_report_native_expression_error(lib):
    ops = backend.JuliacOps(lib)
    lib.expression_ref = 0
    with pytest.raises(RuntimeError, match="expression"):
        ops.apply("sin", [123])
    ops.free()


def test_compiled_variables_return_opaque_array_ref(lib):
    ops = backend.JuliacOps(lib)
    assert ops.add_variables(3) == lib.expression_ref
    assert lib.calls == [("variables", 99, 3)]
    ops.free()


def test_compiled_value_queries_use_expression_ref(lib):
    ops = backend.JuliacOps(lib)
    assert ops.value(123) == 1.25
    lib.value_result = 2.5
    assert ops.value(123) == 2.5
    assert lib.calls == [("value", 99, 123)] * 2
    ops.free()


def test_compiled_value_reports_native_error(lib):
    ops = backend.JuliacOps(lib)
    lib.value_result = float("nan")
    with pytest.raises(RuntimeError, match="value"):
        ops.value(123)
    ops.free()


@pytest.mark.parametrize(
    ("value", "constructor"),
    [(2, "integer_constant"), (2.0, "constant"), (-(2**63), "integer_constant"),
     (2**63 - 1, "integer_constant")],
)
def test_compiled_constants_preserve_integer_and_float_types(lib, value, constructor):
    ops = backend.JuliacOps(lib)
    assert ops.constant(value) == lib.expression_ref
    assert lib.calls == [(constructor, 99, value)]
    assert type(lib.calls[0][-1]) is type(value)
    ops.free()


@pytest.mark.parametrize(
    ("values", "constructor"),
    [([0, 1], "integer_iterator"), ([], "integer_iterator"),
     ([0.0, 1.0], "iterator"), ([0, 1.5], "iterator"),
     ([-(2**63), 2**63 - 1], "integer_iterator")],
)
def test_compiled_iterators_preserve_integer_values(lib, values, constructor):
    ops = backend.JuliacOps(lib)
    assert ops.iterator(values) == lib.expression_ref
    assert lib.calls == [(constructor, 99, tuple(values))]
    expected_type = int if constructor == "integer_iterator" else float
    assert all(type(value) is expected_type for value in lib.calls[0][-1])
    ops.free()


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_compiled_integer_overflow_is_rejected_before_ffi(lib, value):
    ops = backend.JuliacOps(lib)
    with pytest.raises(OverflowError, match="Int64"):
        ops.constant(value)
    with pytest.raises(OverflowError, match="Int64"):
        ops.iterator([value])
    assert lib.calls == []
    ops.free()


def test_compiled_ops_reject_foreign_or_non_native_set(lib):
    ops = backend.JuliacOps(lib)
    with NativeSet(FakeLib(), 1) as foreign:
        for set_ in (foreign, object(), 123):
            with pytest.raises(TypeError, match="jumpy.highs"):
                ops.add_constraint(123, set_)
    assert lib.calls == []
    ops.free()


def test_compiled_ops_reject_closed_set_before_calling_julia(lib):
    ops = backend.JuliacOps(lib)
    set_ = jp.MOI.LessThan(1.0)
    set_.close()
    before = list(lib.calls)
    with pytest.raises(RuntimeError, match="closed"):
        ops.add_constraint(123, set_)
    assert lib.calls == before
    ops.free()


def test_compiled_ops_report_native_constraint_error(lib):
    ops = backend.JuliacOps(lib)
    lib.constraint_status = -1
    with jp.MOI.LessThan(1.0) as set_:
        with pytest.raises(RuntimeError, match="constraint"):
            ops.add_constraint(123, set_)
    ops.free()


def test_explicit_constraint_forwards_same_native_set():
    ops = MockOps()
    model = Model(ops)
    x, y = model.variables(2)
    set_ = object()
    model.constraint(x + y, set_)
    assert ops.constraints == [(("+", x.ref, y.ref), set_)]
    assert ops.constraints[0][1] is set_


def test_comparison_and_explicit_forms_forward_the_same_set():
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    comparison = x + 1 <= 2
    model.constraint(comparison)
    model.constraint(comparison.func, comparison.set)
    assert ops.constraints == [(comparison.func.ref, comparison.set)] * 2
    assert all(set_ is comparison.set for _, set_ in ops.constraints)


def test_constraint_group_uses_the_generic_constraint_path():
    ops = MockOps()
    model = Model(ops)
    x = model.variables(3)
    i = model.iterator(range(3))
    comparison = x[i] >= 1
    model.constraint_group(comparison)
    model.constraint(comparison)
    model.constraint(comparison.func, comparison.set)
    assert comparison.set == ("GreaterThan", 1.0)
    assert ops.constraints == [(comparison.func.ref, comparison.set)] * 3
    assert all(set_ is comparison.set for _, set_ in ops.constraints)


@pytest.mark.parametrize("explicit", [False, True])
def test_bare_variable_forwards_native_ref(explicit):
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    comparison = x <= 2.0
    set_ = comparison.set
    if explicit:
        model.constraint(x, set_)
    else:
        model.constraint(comparison)
    assert ops.constraints == [(x.ref, set_)]


@pytest.mark.parametrize("value", [0, 0.0, 2, 2.0])
def test_numeric_constraint_forwards_constant(value):
    ops = MockOps()
    model = Model(ops)
    set_ = object()
    model.constraint(value, set_)
    assert ops.constraints == [(value, set_)]
    assert type(ops.constraints[0][0]) is type(value)


def test_foreign_model_objective_rejected_before_backend_call(monkeypatch):
    model = Model(MockOps())
    other = Model(MockOps())

    def unexpected_objective(*args):
        pytest.fail("A foreign expression must not reach the native backend")

    monkeypatch.setattr(model._ops, "set_objective", unexpected_objective, raising=False)
    with pytest.raises(ValueError, match="different models"):
        model.objective = jp.minimize(other.variable())


def test_foreign_model_expression_rejected_in_both_constraint_forms():
    ops = MockOps()
    model = Model(ops)
    other = Model(MockOps())
    x = other.variable()
    with pytest.raises(ValueError, match="different model"):
        model.constraint(x, object())
    with pytest.raises(ValueError, match="different model"):
        model.constraint(x <= 1.0)
    assert not ops.constraints


def test_foreign_models_cannot_be_combined_before_constraint_creation():
    first, second = Model(MockOps()), Model(MockOps())
    x = first.variable()
    y = second.variable()
    with pytest.raises(ValueError, match="different models"):
        _ = x + y


@pytest.mark.parametrize("explicit", [False, True])
def test_closed_model_rejects_constraint(explicit):
    ops = MockOps()
    model = Model(ops)
    x = model.variable()
    comparison = x <= 1.0
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        if explicit:
            model.constraint(x, object())
        else:
            model.constraint(comparison)
    assert not ops.constraints


@pytest.mark.parametrize("value", [object(), [], "x", None])
def test_invalid_function_type_rejected_before_backend_call(value):
    ops = MockOps()
    model = Model(ops)
    with pytest.raises(TypeError):
        model.constraint(value, object())
    assert not ops.constraints
