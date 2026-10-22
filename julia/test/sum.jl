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
        for (template, F) in (
            (x[i], MOI.VariableIndex),
            (2x[i] + 3, MOI.ScalarAffineFunction{Float64}),
            (x[i]^2, MOI.ScalarQuadraticFunction{Float64}),
            (sin(x[i]), MOI.ScalarNonlinearFunction),
        )
            total = J.apply(:lazy_sum, Any[template])
            @test total isa GenOpt.LazySum
            @test total.expr === template.expr
            @test total.iterators === template.iterators
            function_ = JuMP.moi_function(total)
            @test function_ isa GenOpt.SumGenerator{F}
            @test length(function_.iterators) == 1
            @test length(only(function_.iterators)) == 2
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
    @test F == GenOpt.SumGenerator{MOI.ScalarQuadraticFunction{Float64}}
    stored = MOI.get(storage, MOI.ObjectiveFunction{F}())
    @test stored.func isa MOI.ScalarNonlinearFunction
    @test length(only(stored.iterators)) == 3
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
          GenOpt.SumGenerator{MOI.ScalarQuadraticFunction{Float64}}
    @test JuMP.moi_function(J._unbox(linear)) isa GenOpt.SumGenerator{MOI.VariableIndex}
    @test length(only(J._unbox(quadratic).iterators)) == 2
    combined = native_apply("+", quadratic, linear)
    @test combined != C_NULL
    moi_func = JuMP.moi_function(J._unbox(combined))
    @test moi_func isa MOI.ScalarNonlinearFunction
    @test all(arg -> arg isa GenOpt.SumGenerator, moi_func.args)
    # A template without variables uses JuMP.VariableRef as its leaf type.
    numeric_sum = native_apply("lazy_sum", native_apply("+", iterator, two))
    @test numeric_sum != C_NULL
    for (head, args) in
        (("+", (numeric_sum, two)), ("*", (two, numeric_sum)), ("sin", (numeric_sum,)))
        expression = native_apply(head, args...)
        @test expression != C_NULL
        @test J._unbox(expression) isa JuMP.NonlinearExpr
        # Pass the result back through the opaque API, not just its producer.
        transformed = native_apply("+", expression, two)
        @test transformed != C_NULL
        @test J._unbox(transformed) isa JuMP.NonlinearExpr
    end
    # HiGHS cannot consume a SumGenerator or a general nonlinear objective.
    @test_throws MOI.UnsupportedAttribute J.JuMPyDirect.set_objective_function(
        J._get(model).model,
        J._unbox(quadratic),
    )
    redirect_stderr(devnull) do
        @test J.jumpy_set_objective_function(model, quadratic) == -1
        @test native_apply("lazy_sum", two) == C_NULL
        @test native_apply("lazy_sum", block) == C_NULL
        @test native_apply("lazy_sum", C_NULL) == C_NULL
    end
    @test J.jumpy_free_model(model) == 0
end
