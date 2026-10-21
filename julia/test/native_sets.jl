@testset "native scalar sets" begin
    J = JuMPyHiGHS
    MOI = J.MOI
    constructors = (
        (() -> J.jumpy_less_than(2.5), MOI.LessThan(2.5)),
        (() -> J.jumpy_greater_than(2.5), MOI.GreaterThan(2.5)),
        (() -> J.jumpy_equal_to(2.5), MOI.EqualTo(2.5)),
        (J.jumpy_zero_one, MOI.ZeroOne()),
        (J.jumpy_integer, MOI.Integer()),
    )
    previous = UInt64(0)
    for (constructor, native) in constructors
        id = constructor()
        @test id > previous
        previous = id
        GC.gc(true)
        @test J._get_set(id) === native
        @test J.jumpy_free_set(id) == 0
        @test J.jumpy_free_set(id) == -1
        @test_throws ArgumentError J._get_set(id)
    end
    @test J.jumpy_free_set(UInt64(0)) == -1
    @test J.jumpy_free_set(typemax(UInt64)) == -1

    @testset "same set survives models and normalization" begin
        id = J.jumpy_less_than(13.0)
        for shift in (3.0, 4.0)
            model = J.jumpy_new_model()
            @test model != C_NULL
            block = J.jumpy_variables(model, Clonglong(1))
            @test block != C_NULL
            x = variable(model, block, 1)
            f = apply(model, "+", x, constant(model, shift))
            @test J.jumpy_add_constraint(model, f, id) == 0
            @test J._get_set(id).upper == 13.0
            @test set_objective(model, 1, x)
            @test J.jumpy_optimize(model) == 1
            @test value(model, x) ≈ 13.0 - shift
            @test J.jumpy_free_model(model) == 0
            GC.gc(true)
        end
        @test J._get_set(id) === MOI.LessThan(13.0)
        @test J.jumpy_free_set(id) == 0
    end

    @testset "MOI owns added constraints after releasing sets" begin
        cases = (
            (() -> J.jumpy_less_than(4.5), 1, 4.5),
            (() -> J.jumpy_greater_than(4.5), 0, 4.5),
            (() -> J.jumpy_equal_to(4.5), 1, 4.5),
            (J.jumpy_zero_one, 1, 1.0),
            (J.jumpy_integer, 1, 4.0),
        )
        for (constructor, objective_sense, optimum) in cases
            model = J.jumpy_new_model()
            block = J.jumpy_variables(model, Clonglong(1))
            @test block != C_NULL
            x = variable(model, block, 1)
            id = constructor()
            @test J.jumpy_add_constraint(model, x, id) == 0
            @test J.jumpy_free_set(id) == 0
            @test J.jumpy_add_constraint(model, x, id) == -1
            @test J.jumpy_add_constraint(model, x, typemax(UInt64)) == -1
            if constructor === J.jumpy_integer
                @test add_constraint(model, x, MOI.LessThan(4.5)) == 0
            end
            @test set_objective(model, objective_sense, x)
            GC.gc(true)
            @test J.jumpy_optimize(model) == 1
            @test value(model, x) ≈ optimum
            @test J.jumpy_free_model(model) == 0
        end
    end

    @testset "native affine expression remains an affine row" begin
        model = J.jumpy_new_model()
        block = J.jumpy_variables(model, Clonglong(1))
        @test block != C_NULL
        x = variable(model, block, 1)
        @test add_constraint(model, x, MOI.GreaterThan(0.0)) == 0
        id = J.jumpy_greater_than(1.0)
        @test J.jumpy_add_constraint(model, apply(model, "+", x, constant(model, 0.0)), id) == 0
        @test J.jumpy_free_set(id) == 0
        @test set_objective(model, 0, x)
        @test J.jumpy_optimize(model) == 1
        @test value(model, x) ≈ 1.0
        @test J.jumpy_free_model(model) == 0
    end
end
