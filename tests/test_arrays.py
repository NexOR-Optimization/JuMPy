"""Variable arrays, vector functions and module operators, without Julia."""

import pytest

import jumpy
from jumpy._registry import NonlinearOperator
from jumpy.model import Model
from mock_ops import MockOps


def test_constrained_variable_matrix_and_slices():
    ops = MockOps()
    model = Model(ops)
    set_ = ("Partition", 3, 2)
    nodes = model.variables((3, 2), set_, name="nodes")
    assert isinstance(nodes, jumpy.VariableArray)
    assert nodes.shape == (3, 2) and len(nodes) == 3
    assert nodes.ref == ("constrained", 0, (3, 2), set_)
    entry = nodes[2, -1]
    assert isinstance(entry, jumpy.Variable)
    assert entry.ref == ("getindex", nodes.ref, ((3, None), (2, None)))
    column = nodes[:, 1]
    assert column.shape == (3,)
    assert column.ref == ("getindex", nodes.ref, ((1, 3), (2, None)))
    assert nodes[1:, :].shape == (2, 2)
    assert nodes[3:, 0].shape == (0,)
    # A vector slice keeps the getindex of its elements.
    assert column[0].ref == ("getindex", column.ref, 1)
    assert column[1:].ref == ("getindex", column.ref, ((2, 3),))
    with pytest.raises(IndexError, match="indices"):
        nodes[0]
    with pytest.raises(IndexError, match="range"):
        nodes[3, 0]
    with pytest.raises(IndexError, match="step"):
        nodes[::2, 0]
    with pytest.raises(TypeError, match="iterable"):
        list(nodes)


def test_variables_with_a_set_reject_bounds_and_shapes_without_set():
    model = Model(MockOps())
    with pytest.raises(ValueError, match="set"):
        model.variables(3, ("Nonnegatives", 3), lower=0)
    with pytest.raises(ValueError, match="need a set"):
        model.variables((2, 2))
    vector = model.variables(3, ("Nonnegatives", 3))
    assert vector.shape == (3,)


def test_vcat_flattens_scalars_vectors_and_sequences():
    ops = MockOps()
    model = Model(ops)
    t = model.variable()
    nodes = model.variables((2, 2), ("Partition", 2, 2))
    vector = jumpy.vcat(t, 3, [nodes[:, 0], 4.5])
    assert isinstance(vector, jumpy.Node)
    assert vector.ref == ("vcat", t.ref, 3, nodes[:, 0].ref, 4.5)
    with pytest.raises(TypeError, match="at least one"):
        jumpy.vcat(1, 2)
    with pytest.raises(ValueError, match="different models"):
        jumpy.vcat(t, Model(MockOps()).variable())


def test_operator_passes_data_arrays_and_expressions():
    ops = MockOps()
    model = Model(ops)
    nodes = model.variables((2, 1), ("Partition", 2, 1))
    route = jumpy.vcat(3, nodes[:, 0], 3)
    distances = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    op = NonlinearOperator("MathOptVRP.op_sum_distances", "sum_distances")
    cost = op(distances, route)
    assert cost.ref == ("sum_distances", ("data", distances), route.ref)
    assert (cost + cost).ref == ("+", cost.ref, cost.ref)
    with pytest.raises(TypeError, match="at least one"):
        op(distances, [3, 1, 3])
