# Requires the experimental JuMP concrete-model patch, not released JuMP.
using JuMP, HiGHS, Test
const MOI = JuMP.MOI

@testset "Concrete backend and native scalar families" begin
    model = @inferred JuMP.concrete_direct_model(HiGHS.Optimizer())
    JuMP.set_silent(model)
    x = @variable(model)
    M, V = typeof(model), typeof(x)
    @test fieldtype(M, :moi_backend) === HiGHS.Optimizer
    @test fieldtype(V, :model) === M
    @test (@inferred backend(model)) isa HiGHS.Optimizer
    @test (@inferred owner_model(x)) === model
    @test (@inferred variable_ref_type(model)) === V
    @test (@inferred value_type(V)) === Float64
    @test (@inferred x + 1) isa GenericAffExpr{Float64,V}
    @test (@inferred x * x) isa GenericQuadExpr{Float64,V}
    @test (@inferred sin(x)) isa GenericNonlinearExpr{V}
    @test all_variables(model) isa Vector{V}
    for expr in (x, 1.0x + 3, x^2 + x + 1)
        func = moi_function(expr)
        @test jump_function_type(model, typeof(func)) === typeof(expr)
        @test jump_function(model, func) isa typeof(expr)
    end
    @constraint(model, x >= 1)
    @objective(model, Min, 1.0x)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test value(x) ≈ 1
    @objective(model, Min, (x - 2)^2)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test value(x) ≈ 2 atol = 1e-6
end

@testset "Legacy public aliases retain concrete types" begin
    model = Model()
    x = @variable(model)
    @test typeof(model) === Model
    @test typeof(x) === VariableRef
    @test x + 1 isa AffExpr
    @test x^2 isa QuadExpr
    @test sin(x) isa NonlinearExpr
    @test VariableRef(model, index(x)) == x
    @test GenericVariableRef{Float64}(model, index(x)) == x
    @test GenericModel{Float32}() isa GenericModel{Float32}
end
