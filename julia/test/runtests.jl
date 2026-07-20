# Tests the C entry points in-process (no compilation needed).

import JuMPyHiGHS
using Test

variable(m, i) = JuMPyHiGHS.jumpy_variable(m, Clonglong(i))
constant(m, v) = JuMPyHiGHS.jumpy_constant(m, Cdouble(v))

function snf(m, head::String, args...)
    argv = collect(Ptr{Cvoid}, args)
    GC.@preserve head argv JuMPyHiGHS.jumpy_scalar_nonlinear(
        m,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

add_constraint(m, f, sense, rhs) =
    JuMPyHiGHS.jumpy_add_constraint(m, f, Cint(sense), Cdouble(rhs))

function set_objective(m, sense, f)
    JuMPyHiGHS.jumpy_set_objective_sense(m, Cint(sense)) == 0 &&
        JuMPyHiGHS.jumpy_set_objective_function(m, f) == 0
end

function get_values(m, n::Int)
    out = zeros(Cdouble, n)
    written = GC.@preserve out JuMPyHiGHS.jumpy_get_values(
        m, pointer(out), Clonglong(n),
    )
    @test written == n
    return out
end

# min x + y  s.t.  x + y >= 10, x, y >= 0
@testset "simple LP" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test m != C_NULL
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(2)) == 0
    # bounds are VariableIndex-in-GreaterThan constraints, as in MOI
    @test add_constraint(m, variable(m, 0), 1, 0.0) >= 0
    @test add_constraint(m, variable(m, 1), 1, 0.0) >= 0
    f = snf(m, "+", variable(m, 0), variable(m, 1))
    @test f != C_NULL
    @test add_constraint(m, f, 1, 10.0) >= 0
    @test set_objective(m, 0, f)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1  # MOI.OPTIMAL
    @test JuMPyHiGHS.jumpy_primal_status(m) == 1  # MOI.FEASIBLE_POINT
    out = get_values(m, 2)
    @test out[1] + out[2] ≈ 10.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_objective_value(m) ≈ 10.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# max x  s.t.  0 <= x <= 42
@testset "maximize with bounds" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(1)) == 0
    @test add_constraint(m, variable(m, 0), 1, 0.0) >= 0
    @test add_constraint(m, variable(m, 0), 0, 42.0) >= 0
    @test set_objective(m, 1, variable(m, 0))
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test get_values(m, 1)[1] ≈ 42.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# equality with a function constant, normalized into the set:
# x + 1 == 5  =>  x == 4
# also exercises simplification: 2 * (x + 3) <= 14  =>  x <= 4
@testset "constant normalization and simplification" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(1)) == 0
    x = variable(m, 0)
    @test add_constraint(m, snf(m, "+", x, constant(m, 1.0)), 2, 5.0) >= 0
    scaled = snf(m, "*", constant(m, 2.0), snf(m, "+", x, constant(m, 3.0)))
    @test add_constraint(m, scaled, 0, 14.0) >= 0
    @test set_objective(m, 0, x)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test get_values(m, 1)[1] ≈ 4.0 atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# constraint group: x[i] >= demand[i] for i in 0..2, demand = (1, 2, 3)
# min sum(x) => x = (1, 2, 3)
@testset "constraint group" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(3)) == 0
    values = Cdouble[0.0, 1.0, 2.0]  # Python-style 0-based iterator values
    demand = Cdouble[1.0, 2.0, 3.0]
    i = GC.@preserve values JuMPyHiGHS.jumpy_iterator(m, pointer(values), Clonglong(3))
    x = JuMPyHiGHS.jumpy_contiguous_variables(m, Clonglong(0), Clonglong(3))
    d = GC.@preserve demand JuMPyHiGHS.jumpy_float_array(m, pointer(demand), Clonglong(3))
    @test i != C_NULL && x != C_NULL && d != C_NULL
    i1 = snf(m, "+", i, constant(m, 1.0))  # 0-based -> 1-based
    template = snf(m, "-", snf(m, "getindex", x, i1), snf(m, "getindex", d, i1))
    n = JuMPyHiGHS.jumpy_add_group_constraint(m, template, Cint(1))
    @test n == 3
    obj = snf(m, "+", variable(m, 0), snf(m, "+", variable(m, 1), variable(m, 2)))
    @test set_objective(m, 0, obj)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test get_values(m, 3) ≈ [1.0, 2.0, 3.0] atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# binary knapsack: max 2 x0 + x1  s.t.  x0 + x1 <= 1, x binary => (1, 0)
@testset "binary variables" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(2)) == 0
    @test add_constraint(m, variable(m, 0), 3, 0.0) >= 0  # ZeroOne
    @test add_constraint(m, variable(m, 1), 3, 0.0) >= 0
    @test add_constraint(m, snf(m, "+", variable(m, 0), variable(m, 1)), 0, 1.0) >= 0
    obj = snf(m, "+", snf(m, "*", constant(m, 2.0), variable(m, 0)), variable(m, 1))
    @test set_objective(m, 1, obj)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test get_values(m, 2) ≈ [1.0, 0.0] atol = 1e-6
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

# `x >= 0` as a *constraint* is a ScalarAffineFunction row (like JuMP's
# @constraint), so it must not clash with an existing variable bound.
@testset "constraint on bounded variable is a row, not a bound" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(1)) == 0
    x = variable(m, 0)
    @test add_constraint(m, x, 1, 0.0) >= 0  # bound: raw VariableIndex
    # constraint: x - 0 >= 0, simplifies to an affine row, no bound clash
    @test add_constraint(m, snf(m, "-", x, constant(m, 0.0)), 1, 0.0) >= 0
    @test set_objective(m, 0, x)
    @test JuMPyHiGHS.jumpy_optimize(m) == 1
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

@testset "unsupported inputs return -1" begin
    m = JuMPyHiGHS.jumpy_new_model()
    @test JuMPyHiGHS.jumpy_add_variables(m, Clonglong(1)) == 0
    redirect_stderr(devnull) do
        # sin(x) <= 1 stays a ScalarNonlinearFunction, which the raw
        # HiGHS.Optimizer does not support
        @test add_constraint(m, snf(m, "sin", variable(m, 0)), 0, 1.0) == -1
        # NULL pointers (any other invalid pointer is undefined behavior,
        # as in any C API)
        @test JuMPyHiGHS.jumpy_optimize(C_NULL) == -1
        @test isnan(JuMPyHiGHS.jumpy_objective_value(C_NULL))
    end
    @test JuMPyHiGHS.jumpy_free_model(m) == 0
end

println("All JuMPyHiGHS tests passed.")
