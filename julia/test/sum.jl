using Test
import JuMPyHiGHS
import JuMP
import GenOpt
import MathOptInterface as MOI

@testset "native lazy sums preserve templates" begin
    J = JuMPyHiGHS.JuMPyModel
    for builder in (J, JuMPyHiGHS.JuMPyDirect)
        model = builder.model(JuMPyHiGHS.Optimizer())
        x = J.add_variables(model, 2)
        i = J.iterator([1, 2])
        for (template, F, G) in (
            (x[i], MOI.ScalarAffineFunction{Float64}, MOI.VariableIndex),
            (
                2x[i] + 3,
                MOI.ScalarAffineFunction{Float64},
                MOI.ScalarAffineFunction{Float64},
            ),
            (
                x[i]^2,
                MOI.ScalarQuadraticFunction{Float64},
                MOI.ScalarQuadraticFunction{Float64},
            ),
            (sin(x[i]), nothing, MOI.ScalarNonlinearFunction),
        )
            total = J.apply(:lazy_sum, Any[template])
            @test total isa GenOpt.LazySum
            @test total.expr === template.expr
            @test total.iterators === template.iterators
            function_ = JuMP.moi_function(total)
            raw = if F === nothing
                function_
            else
                @test function_ isa GenOpt.PolynomialFunction{F}
                @test function_.func.head == :+
                only(function_.func.args)
            end
            @test raw isa GenOpt.SumGenerator{G}
            @test length(raw.iterators) == 1
            @test length(only(raw.iterators)) == 2
        end
    end
end

@testset "lazy objective reaches the MOI model without expansion" begin
    J = JuMPyHiGHS.JuMPyModel
    storage = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    model = JuMP.direct_model(storage)
    x = J.add_variables(model, 3)
    i = J.iterator([1, 2, 3])
    total = J.lazy_sum((x[i] - i)^2)
    J.set_objective_sense(model, MOI.MIN_SENSE)
    J.set_objective_function(model, total)
    F = MOI.get(storage, MOI.ObjectiveFunctionType())
    @test F == GenOpt.PolynomialFunction{MOI.ScalarQuadraticFunction{Float64}}
    stored = MOI.get(storage, MOI.ObjectiveFunction{F}())
    @test stored.func isa MOI.ScalarNonlinearFunction
    raw = only(stored.func.args)
    @test raw isa GenOpt.SumGenerator{MOI.ScalarQuadraticFunction{Float64}}
    @test length(only(raw.iterators)) == 3
end

@testset "lazy LP and QP objectives use GenOpt lowering" begin
    J = JuMPyHiGHS.JuMPyModel
    for builder in (J, JuMPyHiGHS.JuMPyDirect)
        model = builder.model(JuMPyHiGHS.Optimizer())
        x = builder.add_variables(model, 3)
        i = J.iterator([1, 2, 3])
        group = builder.add_constraint(model, x[i] - i, MOI.GreaterThan(0.0))
        total = J.lazy_sum(x[i] + 1)
        row = builder.add_constraint(model, total, MOI.LessThan(12.0))
        if builder === J
            @test JuMP.index(group) isa MOI.ConstraintIndex{
                GenOpt.FunctionGenerator{MOI.ScalarAffineFunction{Float64}},
                MOI.Nonnegatives,
            }
            @test JuMP.index(row) isa MOI.ConstraintIndex{
                GenOpt.PolynomialFunction{MOI.ScalarAffineFunction{Float64}},
                MOI.LessThan{Float64},
            }
        else
            @test group isa Vector{
                MOI.ConstraintIndex{
                    MOI.ScalarAffineFunction{Float64},
                    MOI.GreaterThan{Float64},
                },
            }
            @test length(group) == 3
            @test row isa MOI.ConstraintIndex{
                MOI.ScalarAffineFunction{Float64},
                MOI.LessThan{Float64},
            }
            backend = JuMP.backend(model)
            @test MOI.get(backend, MOI.ConstraintSet(), row) == MOI.LessThan(9.0)
            @test MOI.get(backend, MOI.ConstraintFunction(), row).constant == 0.0
        end
        builder.set_objective_sense(model, MOI.MIN_SENSE)
        builder.set_objective_function(model, J.lazy_sum(i * x[i]))
        builder.optimize(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test builder.value.(x) ≈ [1, 2, 3]
        @test JuMP.objective_value(model) ≈ 14.0

        # Replacing the affine objective exercises quadratic lowering.
        builder.set_objective_function(model, J.lazy_sum((x[i] - i)^2))
        builder.optimize(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test builder.value.(x) ≈ [1, 2, 3] atol = 1e-3
        @test JuMP.objective_value(model) ≈ 0.0 atol = 1e-6
    end
end

@testset "lazy objective lowering preserves linear integer models" begin
    J = JuMPyHiGHS.JuMPyModel
    for builder in (J, JuMPyHiGHS.JuMPyDirect)
        model = builder.model(JuMPyHiGHS.Optimizer())
        x = builder.add_variables(model, 3)
        for variable in x
            builder.add_constraint(model, variable, MOI.ZeroOne())
        end
        i = J.iterator([1, 2, 3])
        builder.add_constraint(model, J.lazy_sum(x[i]), MOI.LessThan(2.0))
        builder.set_objective_sense(model, MOI.MAX_SENSE)
        for objective in (x[1] + 2x[2] + 3x[3], J.lazy_sum(i * x[i]))
            builder.set_objective_function(model, objective)
            builder.optimize(model)
            @test JuMP.termination_status(model) == MOI.OPTIMAL
            @test builder.value.(x) ≈ [0, 1, 1]
            @test JuMP.objective_value(model) ≈ 5.0
        end
    end
end

@testset "lazy sums use the opaque generic operator entry point" begin
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
    indices = Clonglong[1, 2]
    iterator = GC.@preserve indices J.jumpy_integer_iterator(
        model,
        pointer(indices),
        Clonglong(length(indices)),
    )
    template = native_apply("getindex", block, iterator)
    two = J.jumpy_integer_constant(model, Clonglong(2))
    quadratic_template = native_apply("^", template, two)
    quadratic = native_apply("lazy_sum", quadratic_template)
    linear = native_apply("lazy_sum", template)
    @test quadratic != C_NULL
    @test linear != C_NULL
    GC.gc(true)
    @test J._unbox(quadratic) isa J.NativeLazySum
    @test JuMP.moi_function(J._unbox(quadratic)) isa
          GenOpt.PolynomialFunction{MOI.ScalarQuadraticFunction{Float64}}
    @test JuMP.moi_function(J._unbox(linear)) isa
          GenOpt.PolynomialFunction{MOI.ScalarAffineFunction{Float64}}
    @test length(only(J._unbox(quadratic).iterators)) == 2
    combined = native_apply("+", quadratic, linear)
    @test combined != C_NULL
    moi_func = JuMP.moi_function(J._unbox(combined))
    @test moi_func isa GenOpt.PolynomialFunction{MOI.ScalarQuadraticFunction{Float64}}
    # A template without variables uses JuMP.VariableRef as its leaf type.
    numeric_sum = native_apply("lazy_sum", native_apply("+", iterator, two))
    @test numeric_sum != C_NULL
    for (head, args, F) in (
        ("+", (numeric_sum, two), GenOpt.PolynomialExpr),
        ("*", (two, numeric_sum), GenOpt.PolynomialExpr),
        ("sin", (numeric_sum,), JuMP.NonlinearExpr),
    )
        expression = native_apply(head, args...)
        @test expression != C_NULL
        @test J._unbox(expression) isa F
        # Pass the result back through the opaque API, not just its producer.
        transformed = native_apply("+", expression, two)
        @test transformed != C_NULL
        @test J._unbox(transformed) isa F
    end
    @test J.jumpy_set_objective_function(model, quadratic) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test J.jumpy_objective_value(model) ≈ 0.0 atol = 1e-6
    @test J.jumpy_set_objective_function(model, combined) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test J.jumpy_objective_value(model) ≈ -0.5 atol = 1e-6
    @test J.jumpy_set_objective_function(model, numeric_sum) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test J.jumpy_objective_value(model) ≈ 7.0
    nonlinear = native_apply("lazy_sum", native_apply("sin", template))
    redirect_stderr(devnull) do
        @test J.jumpy_set_objective_function(model, nonlinear) == -1
        @test native_apply("lazy_sum", two) == C_NULL
        @test native_apply("lazy_sum", block) == C_NULL
        @test native_apply("lazy_sum", C_NULL) == C_NULL
    end
    @test J.jumpy_free_model(model) == 0
end
