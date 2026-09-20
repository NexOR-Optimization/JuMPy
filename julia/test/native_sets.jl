@testset "native scalar sets" begin
    J = JuMPyHiGHS
    MOI = J.MOI
    sets = (MOI.LessThan(2.5), MOI.GreaterThan(2.5), MOI.EqualTo(2.5), MOI.ZeroOne(), MOI.Integer())
    previous = UInt64(0)
    for (index, native) in enumerate(sets)
        id = J.jumpy_scalar_set(Cint(index - 1), 2.5)
        @test id > previous
        previous = id
        GC.gc(true)
        @test J._get_set(id) === native
        @test J.jumpy_free_set(id) == 0
        @test J.jumpy_free_set(id) == -1
        @test_throws ArgumentError J._get_set(id)
    end
    @test J.jumpy_scalar_set(Cint(-1), 0.0) == 0
    @test J.jumpy_scalar_set(Cint(5), 0.0) == 0
    @test J.jumpy_free_set(UInt64(0)) == -1
    @test J.jumpy_free_set(typemax(UInt64)) == -1

    @testset "same set survives models and normalization" begin
        id = J.jumpy_scalar_set(Cint(0), 13.0)
        for shift in (3.0, 4.0)
            model = J.jumpy_new_model()
            @test model != C_NULL
            @test J.jumpy_add_variables(model, Clonglong(1)) == 0
            x = variable(model, 0)
            f = snf(model, "+", x, constant(model, shift))
            @test J.jumpy_add_constraint_set(model, f, id) >= 0
            @test J._get_set(id).upper == 13.0
            @test set_objective(model, 1, x)
            @test J.jumpy_optimize(model) == 1
            @test J.jumpy_objective_value(model) ≈ 13.0 - shift
            @test J.jumpy_free_model(model) == 0
            GC.gc(true)
        end
        @test J._get_set(id) === MOI.LessThan(13.0)
        @test J.jumpy_free_set(id) == 0
    end

    @testset "MOI owns added constraints after releasing sets" begin
        for (sense, rhs, optimum) in ((0, 4.5, 4.5), (1, 4.5, 4.5), (2, 4.5, 4.5), (3, 0.0, 1.0), (4, 0.0, 4.0))
            model = J.jumpy_new_model()
            @test J.jumpy_add_variables(model, Clonglong(1)) == 0
            x = variable(model, 0)
            id = J.jumpy_scalar_set(Cint(sense), rhs)
            @test J.jumpy_add_constraint_set(model, x, id) >= 0
            @test J.jumpy_free_set(id) == 0
            @test J.jumpy_add_constraint_set(model, x, id) == -1
            @test J.jumpy_add_constraint_set(model, x, typemax(UInt64)) == -1
            if sense == 4
                @test add_constraint(model, x, 0, 4.5) >= 0
            end
            @test set_objective(model, sense == 1 ? 0 : 1, x)
            GC.gc(true)
            @test J.jumpy_optimize(model) == 1
            @test J.jumpy_objective_value(model) ≈ optimum
            @test J.jumpy_free_model(model) == 0
        end
    end

    @testset "nonlinear expression remains an affine row" begin
        model = J.jumpy_new_model()
        @test J.jumpy_add_variables(model, Clonglong(1)) == 0
        x = variable(model, 0)
        @test add_constraint(model, x, 1, 0.0) >= 0
        id = J.jumpy_scalar_set(Cint(1), 1.0)
        @test J.jumpy_add_constraint_set(model, snf(model, "+", x, constant(model, 0.0)), id) >= 0
        @test J.jumpy_free_set(id) == 0
        @test set_objective(model, 0, x)
        @test J.jumpy_optimize(model) == 1
        @test J.jumpy_objective_value(model) ≈ 1.0
        @test J.jumpy_free_model(model) == 0
    end
end
