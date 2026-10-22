using Test
import JuMPyHiGHS
import JuMP
import MathOptInterface as MOI

@testset "scalar constraint duals" begin
    for J in (JuMPyHiGHS.JuMPyModel, JuMPyHiGHS.JuMPyDirect)
        for (sense, set, expected) in (
            (MOI.MIN_SENSE, MOI.GreaterThan(2.0), 1.0),
            (MOI.MAX_SENSE, MOI.LessThan(2.0), -1.0),
            (MOI.MIN_SENSE, MOI.EqualTo(2.0), 1.0),
            (MOI.MAX_SENSE, MOI.EqualTo(2.0), -1.0),
        )
            model = J.model(JuMPyHiGHS.Optimizer())
            x = only(J.add_variables(model, 1))
            row = J.add_constraint(model, x + 0.0, set)
            J.set_objective_sense(model, sense)
            J.set_objective_function(model, x)
            @test_throws Exception J.dual(model, row)
            J.optimize(model)
            @test J.dual(model, row) ≈ expected
            J.add_constraint(model, x + 0.0, MOI.LessThan(3.0))
            if JuMP.mode(model) == JuMP.DIRECT
                # Direct models leave result availability to their solver.
                @test J.dual(model, row) ==
                      MOI.get(JuMP.backend(model), MOI.ConstraintDual(), row)
            else
                @test_throws Exception J.dual(model, row)
            end
        end
        model = J.model(JuMPyHiGHS.Optimizer())
        x = only(J.add_variables(model, 1))
        J.add_constraint(model, x, MOI.ZeroOne())
        row = J.add_constraint(model, x + 0.0, MOI.LessThan(1.0))
        J.set_objective_function(model, x)
        J.optimize(model)
        @test !JuMPyHiGHS.JuMPyModel.has_duals(model)
        @test_throws Exception J.dual(model, row)
    end
end

@testset "opaque native constraint references" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    function native_apply(head::String, args...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_apply(
            model,
            Cstring(pointer(head)),
            pointer(argv),
            Clonglong(length(argv)),
        )
    end
    block = J.jumpy_variables(model, Clonglong(2))
    one = J.jumpy_integer_constant(model, Clonglong(1))
    x = native_apply("getindex", block, one)
    set = J.jumpy_greater_than(2.0)
    row = J.jumpy_add_constraint(model, x, set)
    @test row != C_NULL
    @test J._unbox_constraint(J._get(model), row) isa MOI.ConstraintIndex
    @test J.jumpy_free_set(set) == 0
    @test J.jumpy_set_objective_function(model, x) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    GC.gc(true)
    @test J.jumpy_dual(model, row) ≈ 1.0

    other = J.jumpy_new_model()
    indices = Clonglong[1, 2]
    iterator =
        GC.@preserve indices J.jumpy_integer_iterator(model, pointer(indices), Clonglong(2))
    template = native_apply("getindex", block, iterator)
    set = J.jumpy_less_than(5.0)
    group = J.jumpy_add_constraint(model, template, set)
    @test group != C_NULL
    @test J.jumpy_free_set(set) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    redirect_stderr(devnull) do
        @test isnan(J.jumpy_dual(model, group))
        @test isnan(J.jumpy_dual(other, row))
        @test isnan(J.jumpy_dual(model, x))
        @test isnan(J.jumpy_dual(model, C_NULL))
        @test isnan(J.jumpy_dual(C_NULL, row))
        @test native_apply("+", row, x) == C_NULL
    end
    @test J.jumpy_free_model(other) == 0
    @test J.jumpy_free_model(model) == 0
end
