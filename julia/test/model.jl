import JuMP
import MathOptInterface as MOI

@testset "shared JuMP model operations" begin
    J = JuMPyHiGHS.JuMPyModel
    model = J.model(JuMPyHiGHS.Optimizer())
    x, y = J.add_variables(model, 2)
    @test model isa JuMP.Model
    @test x isa JuMP.VariableRef
    @test JuMP.owner_model(x) === model

    @testset "native operator types and coefficients" begin
        affine = J.apply(:+, Any[x, y])
        @test affine isa JuMP.AffExpr
        @test JuMP.coefficient(affine, x) === 1.0
        @test JuMP.coefficient(affine, y) === 1.0
        for coefficient in (1, 1.0)
            scaled = J.apply(:*, Any[coefficient, x])
            @test scaled isa JuMP.AffExpr
            @test JuMP.coefficient(scaled, x) === 1.0
        end
        quadratic = J.apply(:*, Any[x, y])
        @test quadratic isa JuMP.QuadExpr
        @test JuMP.coefficient(quadratic, x, y) === 1.0
        squared = J.apply(:^, Any[x, 2])
        @test squared isa JuMP.QuadExpr
        @test JuMP.coefficient(squared, x, x) === 1.0
        @test J.apply(:^, Any[x, 2.0]) isa JuMP.NonlinearExpr
        nonlinear = J.apply(:sin, Any[x])
        @test nonlinear isa JuMP.NonlinearExpr
        @test nonlinear.head === :sin
        @test nonlinear.args == [x]
        @test J.apply(:+, Any[2.0, 3.0]) === 5.0
        @test J.apply(:-, Any[x]) isa JuMP.AffExpr
        @test J.apply(:/, Any[x, 2.0]) isa JuMP.AffExpr
        @test J.contiguous_variables(model, 0, 2) == [x, y]
    end

    @testset "JuMP constraint construction" begin
        bound = J.add_constraint(model, x, MOI.GreaterThan(0.0))
        row = J.add_constraint(model, J.apply(:-, Any[x, 0.0]), MOI.GreaterThan(0.0))
        @test JuMP.index(bound) isa MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}
        @test JuMP.index(row) isa MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
        for (value, set) in ((0.0, MOI.LessThan(1.0)), (2.0, MOI.EqualTo(2.0)))
            con = J.add_constraint(model, value, set)
            @test JuMP.constraint_object(con).func isa JuMP.AffExpr
        end
        for constructor in (MOI.LessThan, MOI.GreaterThan, MOI.EqualTo)
            func = J.apply(:+, Any[x, 3.0])
            set = constructor(5.0)
            con = JuMP.constraint_object(J.add_constraint(model, func, set))
            @test con.set == constructor(2.0)
            @test JuMP.constant(con.func) == 0.0
            @test JuMP.coefficient(con.func, x) === 1.0
            @test JuMP.constant(func) == 3.0
            @test set == constructor(5.0)
        end
    end
end

@testset "native quadratic and integer-coefficient generators" begin
    J = JuMPyHiGHS.JuMPyModel
    G = JuMPyHiGHS.GenOpt
    storage = MOI.Utilities.Model{Float64}()
    backend = MOI.Bridges.full_bridge_optimizer(storage, Float64)
    MOI.Bridges.add_bridge(backend, G.FunctionGeneratorBridge{Float64})
    model = JuMP.direct_model(backend)
    x = J.add_variables(model, 3)
    i = J.iterator([1, 2, 3])
    indexed = J.apply(:getindex, Any[x, i])
    square = J.apply(:*, Any[indexed, indexed])
    constraint = J.add_constraint(model, square, MOI.LessThan(9.0))
    @test JuMP.index(constraint) isa MOI.ConstraintIndex{
        G.FunctionGenerator{MOI.ScalarQuadraticFunction{Float64}},
        MOI.Nonpositives,
    }
    @test MOI.get(storage, MOI.NumberOfConstraints{
        MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64},
    }()) == 3
    product = J.apply(:*, Any[
        J.apply(:+, Any[indexed, 1]), J.apply(:-, Any[indexed, 2]),
    ])
    J.add_constraint(model, product, MOI.LessThan(9.0))
    @test MOI.get(storage, MOI.NumberOfConstraints{
        MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64},
    }()) == 6
    for constructor in (MOI.LessThan, MOI.GreaterThan, MOI.EqualTo)
        affine = J.apply(:+, Any[indexed, 0])
        constraint = J.add_constraint(model, affine, constructor(1.0))
        @test JuMP.index(constraint) isa MOI.ConstraintIndex{
            G.FunctionGenerator{MOI.ScalarAffineFunction{Float64}},
        }
    end
end
