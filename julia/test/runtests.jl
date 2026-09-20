# Tests the C entry points in-process (no compilation needed).

import JuMPyHiGHS
using Test

include("model.jl")

variable(m, block, i) = apply(m, "getindex", block, constant(m, i))
value(m, func) = JuMPyHiGHS.jumpy_value(m, func)
constant(m, v) = JuMPyHiGHS.jumpy_constant(m, Cdouble(v))
constant(m, v::Integer) = JuMPyHiGHS.jumpy_integer_constant(m, Clonglong(v))

function apply(m, head::String, args...)
    argv = collect(Ptr{Cvoid}, args)
    GC.@preserve head argv JuMPyHiGHS.jumpy_apply(
        m,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

function add_constraint(m, f, set::MOI.AbstractScalarSet)
    id = JuMPyHiGHS._store_set(set)
    try
        return JuMPyHiGHS.jumpy_add_constraint(m, f, id)
    finally
        JuMPyHiGHS.jumpy_free_set(id)
    end
end

function set_objective(m, sense, f)
    JuMPyHiGHS.jumpy_set_objective_sense(m, Cint(sense)) == 0 &&
        JuMPyHiGHS.jumpy_set_objective_function(m, f) == 0
end

include("native_sets.jl")

@testset "opaque variable arrays and reference-based values" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    first = J.jumpy_variables(model, Clonglong(2))
    second = J.jumpy_variables(model, Clonglong(1))
    empty = J.jumpy_variables(model, Clonglong(0))
    @test J._unbox(first) isa Vector{JuMP.VariableRef}
    @test length(J._unbox(first)) == 2
    @test isempty(J._unbox(empty))
    x = variable(model, first, 1)
    y = variable(model, second, 1)
    @test J._unbox(x) === J._unbox(first)[1]
    @test J._unbox(y) === J._unbox(second)[1]
    @test J._unbox(x) != J._unbox(y)
    @test add_constraint(model, x, MOI.EqualTo(2.0)) == 0
    @test add_constraint(model, y, MOI.EqualTo(3.0)) == 0
    sum = apply(model, "+", x, y)
    product = apply(model, "*", x, y)
    nonlinear = apply(model, "sin", x)
    @test set_objective(model, 0, sum)
    redirect_stderr(devnull) do
        @test isnan(value(model, x))  # no solution yet
        @test variable(model, first, 0) == C_NULL
        @test variable(model, first, 3) == C_NULL
        @test isnan(value(C_NULL, x))
        @test isnan(value(model, C_NULL))
    end
    GC.gc(true)
    @test J.jumpy_optimize(model) == 1
    @test value(model, x) ≈ 2.0
    @test value(model, y) ≈ 3.0
    @test value(model, sum) ≈ 5.0
    @test value(model, product) ≈ 6.0
    @test value(model, nonlinear) ≈ sin(2.0)
    @test J.jumpy_free_model(model) == 0
end

# min x + y  s.t.  x + y >= 10, x, y >= 0
@testset "simple LP" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test m != C_NULL
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(2))
    @test block != C_NULL
    # bounds are VariableIndex-in-GreaterThan constraints, as in MOI
    @test add_constraint(m, variable(m, block, 1), MOI.GreaterThan(0.0)) == 0
    @test add_constraint(m, variable(m, block, 2), MOI.GreaterThan(0.0)) == 0
    f = apply(m, "+", variable(m, block, 1), variable(m, block, 2))
    @test f != C_NULL
    @test add_constraint(m, f, MOI.GreaterThan(10.0)) == 0
    @test set_objective(m, 0, f)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1  # MOI.OPTIMAL
    out = [value(m, variable(m, block, k)) for k in 1:2]
    @test out[1] + out[2] ≈ 10.0 atol = 1e-6
    @test value(m, f) ≈ 10.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# max x  s.t.  0 <= x <= 42
@testset "maximize with bounds" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(1))
    @test block != C_NULL
    @test add_constraint(m, variable(m, block, 1), MOI.GreaterThan(0.0)) == 0
    @test add_constraint(m, variable(m, block, 1), MOI.LessThan(42.0)) == 0
    @test set_objective(m, 1, variable(m, block, 1))
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test value(m, variable(m, block, 1)) ≈ 42.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# equality with a function constant, normalized into the set:
# x + 1 == 5  =>  x == 4
# also exercises simplification: 2 * (x + 3) <= 14  =>  x <= 4
@testset "constant normalization and simplification" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(1))
    @test block != C_NULL
    x = variable(m, block, 1)
    @test add_constraint(m, apply(m, "+", x, constant(m, 1.0)), MOI.EqualTo(5.0)) == 0
    scaled = apply(m, "*", constant(m, 2.0), apply(m, "+", x, constant(m, 3.0)))
    @test add_constraint(m, scaled, MOI.LessThan(14.0)) == 0
    @test set_objective(m, 0, x)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test value(m, x) ≈ 4.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

@testset "opaque JuMP expression types and quadratic solve" begin
    J = JuMPyHiGHS
    m = J.jumpy_new_model()
    block = J.jumpy_variables(m, Clonglong(1))
    @test block != C_NULL
    x = variable(m, block, 1)
    @test J._unbox(x) isa JuMP.VariableRef
    @test J._unbox(constant(m, 1)) === Int64(1)
    @test J._unbox(constant(m, 1.0)) === 1.0
    affine = apply(m, "*", constant(m, 1), x)
    @test J._unbox(affine) isa JuMP.AffExpr
    @test JuMP.coefficient(J._unbox(affine), J._unbox(x)) === 1.0
    square = apply(m, "^", x, constant(m, 2))
    @test J._unbox(square) isa JuMP.QuadExpr
    @test J._unbox(apply(m, "*", x, x)) isa JuMP.QuadExpr
    @test J._unbox(apply(m, "sin", x)) isa JuMP.NonlinearExpr
    objective = apply(m, "+", apply(m, "-", square, apply(m, "*", constant(m, 4), x)), constant(m, 4))
    @test set_objective(m, 0, objective)
    GC.gc(true)
    @test J.jumpy_optimize(m) == 1
    @test value(m, x) ≈ 2.0 atol = 1e-5
    @test value(m, objective) ≈ 0.0 atol = 1e-5
    @test J.jumpy_free_model(m) == 0
end

# constraint group: x[i] >= demand[i] for i in 0..2, demand = (1, 2, 3)
# min sum(x) => x = (1, 2, 3)
@testset "constraint group" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(3))
    @test block != C_NULL
    for k in 0:2
        @test add_constraint(m, variable(m, block, k+1), MOI.GreaterThan(0.0)) == 0
    end
    values = Clonglong[0, 1, 2]  # Python-style 0-based iterator values
    demand = Cdouble[1.0, 2.0, 3.0]
    i = GC.@preserve values JuMPyHiGHS.jumpy_integer_iterator(m, pointer(values), Clonglong(3))
    x = block
    d = GC.@preserve demand JuMPyHiGHS.jumpy_float_array(m, pointer(demand), Clonglong(3))
    @test i != C_NULL && x != C_NULL && d != C_NULL
    i1 = apply(m, "+", i, constant(m, 1))  # 0-based -> 1-based
    template = apply(m, "-", apply(m, "getindex", x, i1), apply(m, "getindex", d, i1))
    @test JuMPyHiGHS._unbox(template) isa JuMPyHiGHS.GenOpt.ExprTemplate{JuMP.AffExpr}
    set = JuMPyHiGHS.jumpy_greater_than(0.0)
    status = JuMPyHiGHS.jumpy_add_constraint(m, template, set)
    @test JuMPyHiGHS.jumpy_free_set(set) == 0
    @test JuMPyHiGHS.jumpy_add_constraint(m, template, set) == -1
    @test status == 0
    # GenOpt's JuMP build_constraint shifts the set into the expression;
    # its function-generator type records the resulting affine row.
    direct = apply(m, "getindex", x, i1)
    @test JuMPyHiGHS._unbox(direct) isa JuMPyHiGHS.GenOpt.ExprTemplate{JuMP.VariableRef}
    generator = JuMPyHiGHS.GenOpt.ExprGenerator(JuMPyHiGHS._unbox(direct))
    @test JuMP.moi_function(generator) isa JuMPyHiGHS.GenOpt.FunctionGenerator{MOI.VariableIndex}
    # Integer literals remain integer until JuMP/GenOpt promotes coefficients.
    affine = apply(m, "+", direct, constant(m, 0))
    for native_set in (MOI.LessThan(4.0), MOI.GreaterThan(1.0))
        @test add_constraint(m, affine, native_set) == 0
    end
    scaled = apply(m, "*", constant(m, 2), direct)
    @test add_constraint(m, scaled, MOI.LessThan(8.0)) == 0
    quadratic = apply(m, "*", direct, direct)
    @test JuMPyHiGHS._unbox(quadratic) isa JuMPyHiGHS.GenOpt.ExprTemplate{JuMP.QuadExpr}
    set = JuMPyHiGHS.jumpy_greater_than(1.0)
    @test JuMPyHiGHS.jumpy_add_constraint(m, direct, set) == 0
    @test JuMPyHiGHS.jumpy_free_set(set) == 0
    obj = apply(m, "+", variable(m, block, 1), apply(m, "+", variable(m, block, 2), variable(m, block, 3)))
    @test set_objective(m, 0, obj)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test [value(m, variable(m, block, k)) for k in 1:3] ≈ [1.0, 2.0, 3.0] atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# binary knapsack: max 2 x0 + x1  s.t.  x0 + x1 <= 1, x binary => (1, 0)
@testset "binary variables" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(2))
    @test block != C_NULL
    @test add_constraint(m, variable(m, block, 1), MOI.ZeroOne()) == 0
    @test add_constraint(m, variable(m, block, 2), MOI.ZeroOne()) == 0
    @test add_constraint(m, apply(m, "+", variable(m, block, 1), variable(m, block, 2)), MOI.LessThan(1.0)) == 0
    obj = apply(m, "+", apply(m, "*", constant(m, 2.0), variable(m, block, 1)), variable(m, block, 2))
    @test set_objective(m, 1, obj)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test [value(m, variable(m, block, k)) for k in 1:2] ≈ [1.0, 0.0] atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# An explicit expression `x - 0 >= 0` is an affine row, so it must not
# clash with an existing variable bound.
@testset "constraint on bounded variable is a row, not a bound" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(1))
    @test block != C_NULL
    x = variable(m, block, 1)
    @test add_constraint(m, x, MOI.GreaterThan(0.0)) == 0  # bound: raw VariableIndex
    # constraint: x - 0 >= 0, simplifies to an affine row, no bound clash
    @test add_constraint(m, apply(m, "-", x, constant(m, 0.0)), MOI.GreaterThan(0.0)) == 0
    @test set_objective(m, 0, x)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

@testset "unsupported inputs return -1" begin
    m = JuMPyHiGHS.jumpy_new_model()
    block = JuMPyHiGHS.jumpy_variables(m, Clonglong(1))
    @test block != C_NULL
    redirect_stderr(devnull) do
        # sin(x) <= 1 produces a ScalarNonlinearFunction, which the
        # HiGHS.Optimizer does not support
        @test add_constraint(m, apply(m, "sin", variable(m, block, 1)), MOI.LessThan(1.0)) == -1
        # NULL pointers (any other invalid pointer is undefined behavior,
        # as in any C API)
        @test JuMPyHiGHS.jumpy_optimize(C_NULL) == -1
        @test isnan(value(C_NULL, C_NULL))
    end
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

println("All JuMPyHiGHS tests passed.")
