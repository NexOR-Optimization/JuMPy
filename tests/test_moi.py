"""Opaque native MOI constructors, without Julia or a compiled library.

The fake library records ABI arguments; it deliberately does not implement
Python equivalents of the Julia sets or function types.
"""

import builtins
import ctypes
import gc
import weakref

import pytest

from jumpy import backend, moi


class _FakeFunction:
    def __init__(self, lib, name):
        self.lib = lib
        self.name = name

    def __call__(self, *args):
        if self.name == "jumpy_moi_scalar_nonlinear":
            head, handles, count = args
            assert isinstance(handles, ctypes.Array)
            assert handles._type_ is ctypes.c_uint64
            args = (head, tuple(handles[:count]), count)
        self.lib.calls.append((self.name, args))
        if self.name == "jumpy_moi_release":
            return self.lib.release_status
        if self.name in self.lib.fail:
            return 0
        self.lib.next_handle += 1
        return self.lib.next_handle


class FakeLib:
    def __init__(self):
        self.calls = []
        self.fail = set()
        self.release_status = 0
        # Exercise values that would be truncated by a 32-bit handle ABI.
        self.next_handle = 2**40
        for name in (
            "jumpy_moi_scalar_set",
            "jumpy_moi_vector_set",
            "jumpy_moi_constant",
            "jumpy_moi_variable",
            "jumpy_moi_scalar_nonlinear",
            "jumpy_moi_release",
        ):
            setattr(self, name, _FakeFunction(self, name))


@pytest.fixture
def constructors():
    lib = FakeLib()
    return moi.MOIConstructors(lib), lib


def test_factory_uses_default_library(monkeypatch):
    lib = FakeLib()
    monkeypatch.setattr(backend, "_load_lib", lambda: lib)
    with moi.MOIConstructors().constant(2.5) as node:
        assert node.handle == 2**40 + 1
    assert lib.calls == [
        ("jumpy_moi_constant", (2.5,)),
        ("jumpy_moi_release", (2**40 + 1,)),
    ]


@pytest.mark.parametrize(
    ("sense", "code"),
    [("<=", 0), (">=", 1), ("==", 2), ("binary", 3), ("integer", 4)],
)
def test_scalar_set_calls_native_constructor(constructors, sense, code):
    factory, lib = constructors
    with factory.scalar_set(sense, 1.5) as node:
        assert isinstance(node, moi.MOIObject)
        assert node.handle > 2**32
        assert lib.calls == [("jumpy_moi_scalar_set", (code, 1.5))]
        # The wrapper is not a duplicate LessThan/GreaterThan/etc. schema.
        assert not hasattr(node, "upper")
        assert not hasattr(node, "lower")
        assert not hasattr(node, "value")


@pytest.mark.parametrize(("sense", "code"), [("<=", 0), (">=", 1), ("==", 2)])
def test_vector_set_calls_native_constructor(constructors, sense, code):
    factory, lib = constructors
    with factory.vector_set(sense, 3) as node:
        assert isinstance(node, moi.MOIObject)
        assert lib.calls == [("jumpy_moi_vector_set", (code, 3))]
        assert not hasattr(node, "dimension")


def test_variable_indices_are_native_one_based(constructors):
    factory, lib = constructors
    with factory.variable(1):
        assert lib.calls == [("jumpy_moi_variable", (1,))]


def test_scalar_nonlinear_passes_uint64_handles(constructors):
    factory, lib = constructors
    with factory.variable(1) as x, factory.constant(2.5) as constant:
        with factory.scalar_nonlinear("+", [x, constant]) as func:
            assert isinstance(func, moi.MOIObject)
            assert lib.calls[-1] == (
                "jumpy_moi_scalar_nonlinear",
                (b"+", (x.handle, constant.handle), 2),
            )
            assert not hasattr(func, "head")
            assert not hasattr(func, "args")


def test_scalar_nonlinear_accepts_empty_arguments(constructors):
    factory, lib = constructors
    with factory.scalar_nonlinear("+", []):
        assert lib.calls == [("jumpy_moi_scalar_nonlinear", (b"+", (), 0))]


def test_close_releases_once_and_invalidates_handle(constructors):
    factory, lib = constructors
    node = factory.constant(2.5)
    handle = node.handle
    node.close()
    node.close()
    assert lib.calls == [
        ("jumpy_moi_constant", (2.5,)),
        ("jumpy_moi_release", (handle,)),
    ]
    with pytest.raises(RuntimeError):
        _ = node.handle


def test_finalizer_releases_unclosed_object(constructors):
    factory, lib = constructors
    node = factory.constant(2.5)
    handle = node.handle
    reference = weakref.ref(node)
    del node
    gc.collect()
    assert reference() is None
    assert lib.calls[-1] == ("jumpy_moi_release", (handle,))


def test_closed_object_cannot_initialize_juliacall(constructors, monkeypatch):
    factory, _ = constructors
    node = factory.constant(2.5)
    node.close()

    def unexpected_julia():
        pytest.fail("A closed object must fail before starting JuliaCall")

    monkeypatch.setattr(factory, "_julia", unexpected_julia)
    with pytest.raises(RuntimeError):
        node.to_julia()


def test_context_manager_releases_on_exception(constructors):
    factory, lib = constructors
    node = factory.constant(1.0)
    handle = node.handle
    with pytest.raises(ValueError, match="example failure"):
        with node as entered:
            assert entered is node
            raise ValueError("example failure")
    assert lib.calls[-1] == ("jumpy_moi_release", (handle,))
    with pytest.raises(RuntimeError):
        _ = node.handle


def test_closed_object_cannot_be_entered_again(constructors):
    factory, _ = constructors
    node = factory.constant(1.0)
    node.close()
    with pytest.raises(RuntimeError):
        with node:
            pytest.fail("A closed object must not enter its context")


def test_release_failure_is_reported(constructors):
    factory, lib = constructors
    node = factory.constant(1.0)
    lib.release_status = -1
    with pytest.raises(RuntimeError):
        node.close()
    # Avoid deliberately injecting a release failure during finalization.
    lib.release_status = 0
    node.close()


def test_scalar_nonlinear_rejects_foreign_library(constructors):
    factory, lib = constructors
    other = moi.MOIConstructors(FakeLib())
    with factory.variable(1) as x, other.constant(2.0) as foreign:
        calls_before = list(lib.calls)
        with pytest.raises(ValueError):
            factory.scalar_nonlinear("+", [x, foreign])
        assert lib.calls == calls_before


def test_factories_using_same_library_can_share_arguments(constructors):
    factory, lib = constructors
    other = moi.MOIConstructors(lib)
    with factory.variable(1) as x, other.constant(2.0) as constant:
        with factory.scalar_nonlinear("+", [x, constant]):
            assert lib.calls[-1] == (
                "jumpy_moi_scalar_nonlinear",
                (b"+", (x.handle, constant.handle), 2),
            )


def test_scalar_nonlinear_rejects_released_argument(constructors):
    factory, lib = constructors
    node = factory.constant(2.0)
    node.close()
    calls_before = list(lib.calls)
    with pytest.raises(RuntimeError):
        factory.scalar_nonlinear("+", [node])
    assert lib.calls == calls_before


def test_scalar_nonlinear_rejects_raw_handle(constructors):
    factory, lib = constructors
    with factory.constant(2.0) as node:
        calls_before = list(lib.calls)
        with pytest.raises(TypeError):
            factory.scalar_nonlinear("+", [node.handle])
        assert lib.calls == calls_before


@pytest.mark.parametrize(
    ("method", "args", "symbol"),
    [
        ("scalar_set", ("<=", 1.0), "jumpy_moi_scalar_set"),
        ("vector_set", ("==", 3), "jumpy_moi_vector_set"),
        ("constant", (2.5,), "jumpy_moi_constant"),
        ("variable", (1,), "jumpy_moi_variable"),
        ("scalar_nonlinear", ("+", []), "jumpy_moi_scalar_nonlinear"),
    ],
)
def test_null_constructor_result_is_reported(constructors, method, args, symbol):
    factory, lib = constructors
    lib.fail.add(symbol)
    with pytest.raises(RuntimeError):
        getattr(factory, method)(*args)
    assert all(name != "jumpy_moi_release" for name, _ in lib.calls)


def test_compiled_constructors_do_not_import_juliacall(constructors, monkeypatch):
    factory, _ = constructors
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "juliacall" or name.startswith("juliacall."):
            pytest.fail("Compiled-only constructors must not import JuliaCall")
        if name == "jumpy.bridge_juliacall":
            pytest.fail("The JuliaCall bridge must be imported lazily")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with factory.scalar_set("<=", 1.0), factory.variable(1) as x:
        with factory.constant(1.0) as constant:
            with factory.scalar_nonlinear("+", [x, constant]):
                pass
