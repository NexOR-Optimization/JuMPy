using Test
import JuMPyHiGHS
import JuMP
import MathOptInterface as MOI

@testset "native model result queries" begin
    for J in (JuMPyHiGHS.JuMPyModel, JuMPyHiGHS.JuMPyDirect)
        @testset "$J" begin
            for quadratic in (false, true)
                model = J.model(JuMPyHiGHS.Optimizer())
                x = only(J.add_variables(model, 1))
                J.add_constraint(model, x, MOI.GreaterThan(0.0))
                J.add_constraint(model, x, MOI.LessThan(5.0))
                J.set_objective_sense(model, MOI.MIN_SENSE)
                J.set_objective_function(model, quadratic ? (x - 2)^2 : x)
                @test J.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
                @test !J.has_values(model)
                @test !J.has_duals(model)
                @test_throws Exception J.objective_value(model)
                J.optimize(model)
                @test J.termination_status(model) == MOI.OPTIMAL
                @test J.has_values(model)
                @test J.has_duals(model)
                @test J.objective_value(model) ≈ 0.0 atol = 1e-10
                J.add_constraint(model, x + 0.0, MOI.LessThan(4.0))
                @test J.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
                @test !J.has_values(model)
                @test !J.has_duals(model)
                @test_throws Exception J.objective_value(model)
            end

            model = J.model(JuMPyHiGHS.Optimizer())
            x = only(J.add_variables(model, 1))
            J.add_constraint(model, x, MOI.ZeroOne())
            J.set_objective_sense(model, MOI.MAX_SENSE)
            J.set_objective_function(model, x)
            J.optimize(model)
            @test J.has_values(model)
            @test !J.has_duals(model)
            @test J.objective_value(model) ≈ 1.0

            model = J.model(JuMPyHiGHS.Optimizer())
            x = only(J.add_variables(model, 1))
            J.add_constraint(model, x, MOI.GreaterThan(1.0))
            J.add_constraint(model, x, MOI.LessThan(0.0))
            J.set_objective_function(model, x)
            J.optimize(model)
            @test J.termination_status(model) == MOI.INFEASIBLE
            @test !J.has_values(model)
            @test !J.has_duals(model)
            @test_throws Exception J.objective_value(model)
        end
    end
end

@testset "C result entry points" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    block = J.jumpy_variables(model, Clonglong(1))
    one = J.jumpy_integer_constant(model, Clonglong(1))
    head = "getindex"
    argv = Ptr{Cvoid}[block, one]
    x = GC.@preserve head argv J.jumpy_apply(
        model,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(2),
    )
    set = J.jumpy_greater_than(2.0)
    @test J.jumpy_add_constraint(model, x, set) == 0
    @test J.jumpy_free_set(set) == 0
    @test J.jumpy_has_values(model) == 0
    @test J.jumpy_has_duals(model) == 0
    @test J.jumpy_termination_status(model) == Int(MOI.OPTIMIZE_NOT_CALLED)
    @test J.jumpy_set_objective_function(model, x) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test J.jumpy_has_values(model) == 1
    @test J.jumpy_has_duals(model) == 1
    @test J.jumpy_objective_value(model) ≈ 2.0
    redirect_stderr(devnull) do
        @test J.jumpy_has_values(C_NULL) == -1
        @test J.jumpy_has_duals(C_NULL) == -1
        @test J.jumpy_termination_status(C_NULL) == -1
        @test isnan(J.jumpy_objective_value(C_NULL))
    end
    @test J.jumpy_free_model(model) == 0
end
