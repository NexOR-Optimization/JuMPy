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
        self.constraint_status = 1

    def jumpy_scalar_set(self, sense, rhs):
        self.calls.append(("set", sense, rhs))
        self.next_handle += 1
        return 0 if self.fail_construction else self.next_handle

    def jumpy_free_set(self, handle):
        self.calls.append(("free_set", handle))
        return self.free_status

    def jumpy_new_model(self):
        return 99

    def jumpy_free_model(self, model):
        return 0

    def jumpy_add_constraint_set(self, model, func, set_handle):
        self.calls.append(("constraint", model, func, set_handle))
        return self.constraint_status


@pytest.fixture
def lib(monkeypatch):
    library = FakeLib()
    monkeypatch.setattr(backend, "_load_lib", lambda: library)
    return library


@pytest.mark.parametrize(
    ("name", "args", "code", "rhs"),
    [
        ("LessThan", (1.5,), 0, 1.5),
        ("GreaterThan", (2.5,), 1, 2.5),
        ("EqualTo", (3.5,), 2, 3.5),
        ("ZeroOne", (), 3, 0.0),
        ("Integer", (), 4, 0.0),
    ],
)
def test_scalar_sets_call_julia_constructors(lib, name, args, code, rhs):
    with getattr(jp.MOI, name)(*args) as set_:
        assert isinstance(set_, NativeSet)
        assert set_.handle > 2**32
        assert lib.calls == [("set", code, rhs)]
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
    assert lib.calls == [("set", 0, 1.0), ("free_set", handle)]
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
    assert lib.calls == [("set", 0, 1.0)]


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
        ops.add_constraint_set(123, set_)
        assert lib.calls[-1] == ("constraint", 99, 123, set_.handle)
    ops.free()


def test_compiled_ops_reject_foreign_or_non_native_set(lib):
    ops = backend.JuliacOps(lib)
    with NativeSet(FakeLib(), 1) as foreign:
        for set_ in (foreign, object(), 123):
            with pytest.raises(TypeError, match="jumpy.highs"):
                ops.add_constraint_set(123, set_)
    assert lib.calls == []
    ops.free()


def test_compiled_ops_reject_closed_set_before_calling_julia(lib):
    ops = backend.JuliacOps(lib)
    set_ = jp.MOI.LessThan(1.0)
    set_.close()
    before = list(lib.calls)
    with pytest.raises(RuntimeError, match="closed"):
        ops.add_constraint_set(123, set_)
    assert lib.calls == before
    ops.free()


def test_compiled_ops_report_native_constraint_error(lib):
    ops = backend.JuliacOps(lib)
    lib.constraint_status = -1
    with jp.MOI.LessThan(1.0) as set_:
        with pytest.raises(RuntimeError, match="native-set constraint"):
            ops.add_constraint_set(123, set_)
    ops.free()


class ExplicitSetOps(MockOps):
    def __init__(self):
        super().__init__()
        self.explicit_constraints = []

    def add_constraint_set(self, func, set_):
        self.explicit_constraints.append((func, set_))


def test_explicit_constraint_forwards_same_native_set():
    ops = ExplicitSetOps()
    model = Model(ops)
    x, y = model.variables(2)
    set_ = object()
    model.constraint(x + y, set_)
    assert ops.explicit_constraints == [(("+", ("var", 0), ("var", 1)), set_)]
    assert ops.explicit_constraints[0][1] is set_


def test_explicit_bare_variable_stays_a_variable_index():
    ops = ExplicitSetOps()
    model = Model(ops)
    x = model.variable()
    set_ = object()
    model.constraint(x, set_)
    assert ops.explicit_constraints == [(("var", 0), set_)]


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_numeric_constraint_is_wrapped_in_a_function(value):
    ops = ExplicitSetOps()
    model = Model(ops)
    set_ = object()
    model.constraint(value, set_)
    assert ops.explicit_constraints == [(("+", value), set_)]


def test_foreign_model_expression_rejected_in_both_constraint_forms():
    ops = ExplicitSetOps()
    model = Model(ops)
    other = Model(ExplicitSetOps())
    x = other.variable()
    with pytest.raises(ValueError, match="different model"):
        model.constraint(x, object())
    with pytest.raises(ValueError, match="different model"):
        model.constraint(x <= 1.0)
    assert not ops.constraints
    assert not ops.explicit_constraints


def test_foreign_models_cannot_be_combined_before_constraint_creation():
    first, second = Model(ExplicitSetOps()), Model(ExplicitSetOps())
    x = first.variable()
    y = second.variable()
    with pytest.raises(ValueError, match="different models"):
        _ = x + y


@pytest.mark.parametrize("explicit", [False, True])
def test_closed_model_rejects_constraint(explicit):
    ops = ExplicitSetOps()
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
    assert not ops.explicit_constraints


@pytest.mark.parametrize("value", [object(), [], "x", None])
def test_invalid_function_type_rejected_before_backend_call(value):
    ops = ExplicitSetOps()
    model = Model(ops)
    with pytest.raises(TypeError):
        model.constraint(value, object())
    assert not ops.explicit_constraints
