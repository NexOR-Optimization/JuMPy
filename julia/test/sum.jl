using Test
import JuMPyHiGHS
import JuMP
import MathOptInterface as MOI

@testset "native expression accumulation" begin
    J = JuMPyHiGHS.JuMPyModel
    model = J.model(JuMPyHiGHS.Optimizer())
    x, y = J.add_variables(model, 2)
    @test J.sum_terms(model, Any[]) == 0
    @test J.sum_terms(model, Any[2, 3.5]) == 5.5
    affine = J.sum_terms(model, Any[x, 2y, 3])
    @test affine isa JuMP.AffExpr
    @test JuMP.coefficient(affine, x) == 1.0
    @test JuMP.coefficient(affine, y) == 2.0
    @test JuMP.constant(affine) == 3.0
    quadratic = J.sum_terms(model, Any[x^2, y, 3])
    @test quadratic isa JuMP.QuadExpr
    @test JuMP.coefficient(quadratic, x, x) == 1.0
    @test JuMP.coefficient(quadratic, y) == 1.0
    @test JuMP.constant(quadratic) == 3.0
    @test J.sum_terms(model, Any[sin(x), y]) isa JuMP.NonlinearExpr
    @test J.sum_terms(model, Any[affine, affine]) == 2affine
    @test JuMP.coefficient(affine, x) == 1.0
end

@testset "opaque native sum entry point" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    handle = J._get(model)
    x, y = J.JuMPyDirect.add_variables(handle.model, 2)
    for variable in (x, y)
        J.JuMPyDirect.add_constraint(handle.model, variable, MOI.GreaterThan(0.0))
        J.JuMPyDirect.add_constraint(handle.model, variable, MOI.LessThan(5.0))
    end
    function native_sum(values)
        nodes = Ptr{Cvoid}[J._box(handle, value) for value in values]
        return GC.@preserve nodes J.jumpy_sum(
            model,
            pointer(nodes),
            Clonglong(length(nodes)),
        )
    end
    empty = J.jumpy_sum(model, Ptr{Ptr{Cvoid}}(C_NULL), Clonglong(0))
    numeric = native_sum([2, 3.5])
    affine = native_sum(Any[x, 2y, 3])
    doubled = native_sum(Any[J._unbox(affine), J._unbox(affine)])
    quadratic = native_sum(Any[(x-2)^2, (y-3)^2])
    nonlinear = native_sum(Any[sin(x), y])
    @test J._unbox(affine) isa J.NativeAffine
    @test J._unbox(quadratic) isa J.NativeQuadratic
    @test J._unbox(nonlinear) isa J.NativeNonlinear
    @test J.jumpy_set_objective_function(model, quadratic) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    for (node, expected) in (
        (empty, 0.0),
        (numeric, 5.5),
        (affine, 11.0),
        (doubled, 22.0),
        (quadratic, 0.0),
        (nonlinear, sin(2.0) + 3.0),
    )
        @test J.jumpy_value(model, node) ≈ expected atol = 2e-6
    end
    redirect_stderr(devnull) do
        @test J.jumpy_sum(model, Ptr{Ptr{Cvoid}}(C_NULL), Clonglong(1)) == C_NULL
        @test J.jumpy_sum(model, Ptr{Ptr{Cvoid}}(C_NULL), Clonglong(-1)) == C_NULL
        @test J.jumpy_sum(C_NULL, Ptr{Ptr{Cvoid}}(C_NULL), Clonglong(0)) == C_NULL
        @test native_sum(Any[[x, y]]) == C_NULL
    end
    @test J.jumpy_free_model(model) == 0
end
