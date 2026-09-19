import MathOptInterface as MOI

@testset "shared MOI constructors" begin
    constructors = JuMPyHiGHS.JuMPyMOI

    @testset "nonlinear functions retain their arguments" begin
        x = MOI.VariableIndex(1)
        args = Any[x, 2.0]
        inner = constructors.scalar_nonlinear(:+, args)
        outer = constructors.scalar_nonlinear(:*, Any[inner, inner])
        @test inner isa MOI.ScalarNonlinearFunction
        @test inner.head === :+
        @test inner.args === args
        @test outer.head === :*
        @test outer.args[1] === inner
        @test outer.args[2] === inner
    end

    @testset "scalar sets" begin
        sets = (
            MOI.LessThan(1.5),
            MOI.GreaterThan(1.5),
            MOI.EqualTo(1.5),
            MOI.ZeroOne(),
            MOI.Integer(),
        )
        for (i, expected) in enumerate(sets)
            actual = constructors.scalar_set(i - 1, 1.5)
            @test typeof(actual) === typeof(expected)
            @test actual == expected
            @test constructors.scalar_set(Val(i - 1), 1.5) == expected
        end
        for sense in (-1, 5)
            @test_throws ErrorException constructors.scalar_set(sense, 1.5)
        end
    end

    @testset "vector sets" begin
        for n in (0, 3)
            sets = (MOI.Nonpositives(n), MOI.Nonnegatives(n), MOI.Zeros(n))
            for (i, expected) in enumerate(sets)
                actual = constructors.vector_set(i - 1, n)
                @test typeof(actual) === typeof(expected)
                @test MOI.dimension(actual) == n
            end
        end
        for sense in (-1, 3)
            @test_throws ErrorException constructors.vector_set(sense, 3)
        end
    end

    @testset "simplification preserves rows" begin
        x = MOI.VariableIndex(1)
        @test constructors.simplify(x) === x
        @test constructors.simplify(2.0) === 2.0

        row = constructors.simplify(constructors.scalar_nonlinear(:-, Any[x, 0.0]))
        @test row isa MOI.ScalarAffineFunction{Float64}
        @test row.constant == 0.0
        @test row.terms == [MOI.ScalarAffineTerm(1.0, x)]

        constant = constructors.simplify(
            constructors.scalar_nonlinear(:+, Any[2.0, 3.0]),
        )
        @test constant isa MOI.ScalarAffineFunction{Float64}
        @test constant.constant == 5.0
        @test isempty(constant.terms)

        # The common helper must not inherit the compiled ABI's restricted
        # function union: JuliaCall can also use quadratic MOI functions.
        quadratic = constructors.simplify(
            constructors.scalar_nonlinear(:*, Any[x, x]),
        )
        @test quadratic isa MOI.ScalarQuadraticFunction{Float64}
    end

    @testset "normalization moves constants into sets" begin
        sets = (MOI.LessThan(2.0), MOI.GreaterThan(2.0), MOI.EqualTo(2.0))
        for (sense, expected) in enumerate(sets)
            model = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(model)
            func = constructors.simplify(constructors.scalar_nonlinear(:+, Any[x, 3.0]))
            ci = constructors.normalize_and_add_constraint(model, func, sense - 1, 5.0)
            @test ci isa MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},typeof(expected)}
            @test MOI.get(model, MOI.ConstraintSet(), ci) == expected
            normalized = MOI.get(model, MOI.ConstraintFunction(), ci)
            @test normalized.constant == 0.0
            @test normalized.terms == [MOI.ScalarAffineTerm(1.0, x)]
            @test func.constant == 3.0
        end
    end

    @testset "variable domains remain distinct from affine rows" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(model)
        bound = constructors.normalize_and_add_constraint(model, x, 1, 0.0)
        func = constructors.simplify(constructors.scalar_nonlinear(:-, Any[x, 0.0]))
        row = constructors.normalize_and_add_constraint(model, func, 1, 0.0)
        @test bound isa MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}
        @test row isa MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
        @test MOI.get(model, MOI.ConstraintFunction(), bound) == x
        @test MOI.get(model, MOI.ConstraintSet(), row) == MOI.GreaterThan(0.0)

        for (sense, expected) in ((3, MOI.ZeroOne()), (4, MOI.Integer()))
            variable = MOI.add_variable(model)
            ci = constructors.normalize_and_add_constraint(model, variable, sense, 0.0)
            @test ci isa MOI.ConstraintIndex{MOI.VariableIndex,typeof(expected)}
            @test MOI.get(model, MOI.ConstraintFunction(), ci) == variable
            @test MOI.get(model, MOI.ConstraintSet(), ci) == expected
        end
        @test_throws ErrorException constructors.normalize_and_add_constraint(model, x, 5, 0.0)
    end
end
