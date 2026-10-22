using Test
import HiGHS
import JuMP
import JuMPyHiGHS
import MathOptInterface as MOI

@testset "time limits in both Julia adapters" begin
    for J in (JuMPyHiGHS.JuMPyModel, JuMPyHiGHS.JuMPyDirect)
        model = J.model(HiGHS.Optimizer())
        @test MOI.get(JuMP.backend(model), MOI.TimeLimitSec()) === nothing
        @test (@inferred J.set_time_limit(model, 10.0)) === nothing
        @test MOI.get(JuMP.backend(model), MOI.TimeLimitSec()) == 10.0
        for seconds in (-1.0, Inf, -Inf)
            @test_throws ArgumentError J.set_time_limit(model, seconds)
        end
        @test MOI.get(JuMP.backend(model), MOI.TimeLimitSec()) == 10.0
        @test (@inferred J.set_time_limit(model, NaN)) === nothing
        @test MOI.get(JuMP.backend(model), MOI.TimeLimitSec()) === nothing
        x = only(J.add_variables(model, 1))
        J.add_constraint(model, x, MOI.GreaterThan(1.0))
        J.set_objective_sense(model, MOI.MIN_SENSE)
        J.set_objective_function(model, x)
        J.set_time_limit(model, 10.0)
        J.optimize(model)
        @test J.value(x) ≈ 1.0
        J.set_time_limit(model, NaN)
        J.optimize(model)
        @test J.value(x) ≈ 1.0
    end
end

@testset "time-limit C entry point" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    try
        backend = JuMP.backend(J._get(model).model)
        @test J.jumpy_set_time_limit(model, 5.0) == 0
        @test MOI.get(backend, MOI.TimeLimitSec()) == 5.0
        @test J.jumpy_set_time_limit(model, 0.0) == 0
        @test MOI.get(backend, MOI.TimeLimitSec()) == 0.0
        @test J.jumpy_set_time_limit(model, NaN) == 0
        @test MOI.get(backend, MOI.TimeLimitSec()) === nothing
        redirect_stderr(devnull) do
            for seconds in (-1.0, Inf, -Inf)
                @test J.jumpy_set_time_limit(model, seconds) == -1
            end
            @test J.jumpy_set_time_limit(C_NULL, 1.0) == -1
        end
    finally
        @test J.jumpy_free_model(model) == 0
    end
end
