using Test
import JuMPyHiGHS
import JuMP
import MathOptInterface as MOI

@testset "bridge-free concrete JuMP adapter" begin
    J = JuMPyHiGHS.JuMPyDirect
    @test JuMPyHiGHS.NativeModel ===
          JuMP.ConcreteModel{Float64,JuMPyHiGHS.Optimizer,Nothing}
    model = J.model(JuMPyHiGHS.Optimizer())
    @test isconcretetype(JuMPyHiGHS.NativeModel)
    @test typeof(model) === JuMPyHiGHS.NativeModel
    @test typeof(model.optimize_hook) === Nothing
    backend = JuMP.backend(model)
    @test backend isa JuMPyHiGHS.Optimizer
    @test JuMP.mode(model) == JuMP.DIRECT
    @test typeof(model).parameters[2] === typeof(backend)
    @test isempty(J.add_variables(model, 0))
    x, y = J.add_variables(model, 2)
    @test JuMP.owner_model(x) === model
    @test J.add_constraint(model, x, MOI.GreaterThan(0.0)) isa
          MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}
    expression = x + 3.0
    row = J.add_constraint(model, expression, MOI.GreaterThan(5.0))
    @test row isa
          MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
    @test MOI.get(backend, MOI.ConstraintSet(), row) == MOI.GreaterThan(2.0)
    @test JuMP.constant(expression) == 3.0
    @test J.add_constraint(model, y, MOI.EqualTo(3.0)) isa MOI.ConstraintIndex
    @test J.add_constraint(model, 0.0, MOI.LessThan(1.0)) isa
          MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}
    J.set_objective_sense(model, MOI.MIN_SENSE)
    J.set_objective_function(model, x)
    @test MOI.get(backend, MOI.ObjectiveFunctionType()) ===
          MOI.ScalarAffineFunction{Float64}
    @test_throws JuMP.OptimizeNotCalled J.value(x)
    J.optimize(model)
    @test J.value(x) ≈ 2.0
    @test J.value(x + y) ≈ 5.0
    @test J.value(x * y) ≈ 6.0
    @test J.value(sin(x)) ≈ sin(2.0)
    @test J.value(2) === 2.0
    J.set_objective_function(model, 4)
    @test_throws JuMP.OptimizeNotCalled J.value(x)
    J.optimize(model)
    @test MOI.get(backend, MOI.ObjectiveValue()) ≈ 4.0

    @test_throws MOI.UnsupportedConstraint J.add_constraint(
        model,
        sin(x),
        MOI.LessThan(1.0),
    )
    @test_throws MOI.UnsupportedAttribute J.set_objective_function(model, sin(x))
    other = J.model(JuMPyHiGHS.Optimizer())
    foreign = only(J.add_variables(other, 1))
    @test_throws JuMP.VariableNotOwned J.add_constraint(
        model,
        x + foreign,
        MOI.LessThan(1.0),
    )
    @test_throws JuMP.VariableNotOwned J.set_objective_function(model, foreign)
    @test_throws JuMP.VariableNotOwned J.set_objective_function(model, x * foreign)

    quadratic = J.model(JuMPyHiGHS.Optimizer())
    z = only(J.add_variables(quadratic, 1))
    J.set_objective_sense(quadratic, MOI.MIN_SENSE)
    J.set_objective_function(quadratic, (z - 2)^2)
    J.optimize(quadratic)
    @test J.value(z) ≈ 2.0 atol = 1e-6
    @test J.value((z - 2)^2) ≈ 0.0 atol = 1e-6
end

@testset "direct native GenOpt expansion" begin
    J = JuMPyHiGHS.JuMPyDirect
    G = JuMPyHiGHS.GenOpt
    model = J.model(JuMPyHiGHS.Optimizer())
    backend = JuMP.backend(model)
    x = J.add_variables(model, 3)
    i = G.iterator([1, 2, 3])
    demand = [1.0, 2.0, 3.0]
    for variable in x
        J.add_constraint(model, variable, MOI.GreaterThan(0.0))
    end
    rows = J.add_constraint(model, x[i] - demand[i], MOI.GreaterThan(0.0))
    @test rows isa Vector{
        MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},
    }
    @test length(rows) == 3
    @test length(J.add_constraint(model, x[i], MOI.LessThan(4.0))) == 3
    equalities = J.add_constraint(model, x[i] - demand[i], MOI.EqualTo(0.0))
    @test equalities isa Vector{
        MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
    }
    @test length(equalities) == 3
    J.set_objective_sense(model, MOI.MIN_SENSE)
    J.set_objective_function(model, sum(x))
    J.optimize(model)
    @test J.value.(x) ≈ demand
    @test MOI.get(
        backend,
        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
    ) == 3
    @test MOI.get(
        backend,
        MOI.NumberOfConstraints{
            MOI.ScalarAffineFunction{Float64},
            MOI.GreaterThan{Float64},
        }(),
    ) == 3
    @test_throws MOI.UnsupportedConstraint J.add_constraint(
        model,
        x[i]^2,
        MOI.LessThan(16.0),
    )
    @test_throws MOI.UnsupportedConstraint J.add_constraint(
        model,
        sin(x[i]),
        MOI.LessThan(1.0),
    )
    other = J.model(JuMPyHiGHS.Optimizer())
    foreign = J.add_variables(other, 3)
    @test_throws JuMP.VariableNotOwned J.add_constraint(
        model,
        x[i] + foreign[i],
        MOI.LessThan(10.0),
    )
end

@testset "checked native expression dispatch" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    @test model != C_NULL
    function apply_native(head::String, args...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_apply(
            model,
            Cstring(pointer(head)),
            pointer(argv),
            Clonglong(length(argv)),
        )
    end
    integer(value) = J.jumpy_integer_constant(model, Clonglong(value))
    block = J.jumpy_variables(model, Clonglong(2))
    x = apply_native("getindex", block, integer(1))
    y = apply_native("getindex", block, integer(2))
    @test x != C_NULL
    @test y != C_NULL
    for (variable, value) in ((x, 2.0), (y, 3.0))
        set = J.jumpy_equal_to(value)
        @test J.jumpy_add_constraint(model, variable, set) == 0
        @test J.jumpy_free_set(set) == 0
    end
    affine = apply_native("*", integer(2), x)
    @test J._unbox(affine) isa J.NativeAffine
    @test JuMP.coefficient(J._unbox(affine), J._unbox(x)) === 2.0
    square = apply_native("^", x, integer(2))
    @test J._unbox(square) isa J.NativeQuadratic
    @test J.jumpy_set_objective_function(model, affine) == 0
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test J.jumpy_value(model, affine) ≈ 4.0
    @test J.jumpy_value(model, square) ≈ 4.0
    for (head, expected) in (
        ("+", 2.0),
        ("-", -2.0),
        ("sin", sin(2.0)),
        ("cos", cos(2.0)),
        ("exp", exp(2.0)),
        ("log", log(2.0)),
        ("sqrt", sqrt(2.0)),
        ("abs", 2.0),
    )
        expression = apply_native(head, x)
        @test expression != C_NULL
        @test J.jumpy_value(model, expression) ≈ expected
    end
    for (head, lhs, rhs, expected) in (
        ("+", x, y, 5.0),
        ("-", x, y, -1.0),
        ("*", x, y, 6.0),
        ("/", x, integer(2), 1.0),
        ("^", x, integer(3), 8.0),
        ("+", affine, square, 8.0),
        ("+", integer(1), J.jumpy_constant(model, 2.0), 3.0),
    )
        expression = apply_native(head, lhs, rhs)
        @test expression != C_NULL
        @test J.jumpy_value(model, expression) ≈ expected
    end
    # This is a valid rooted handle, but its payload is not an expression
    # supported by the compiled backend. Arbitrary invalid pointers are not
    # part of the C ABI contract.
    unsupported = J._box(J._get(model), "not a native expression")
    set = J.jumpy_less_than(10.0)
    redirect_stderr(devnull) do
        @test apply_native("unsupported", x) == C_NULL
        @test apply_native("sin") == C_NULL
        @test apply_native("sin", x, y) == C_NULL
        @test apply_native("/", x) == C_NULL
        @test apply_native("+", x, y, x) == C_NULL
        @test apply_native("getindex", block) == C_NULL
        @test apply_native("*", block, integer(2)) == C_NULL
        @test apply_native("+", unsupported, x) == C_NULL
        @test J.jumpy_add_constraint(model, unsupported, set) == -1
        @test J.jumpy_set_objective_function(model, unsupported) == -1
        @test isnan(J.jumpy_value(model, unsupported))
        @test J.jumpy_add_constraint(model, block, set) == -1
        @test J.jumpy_set_objective_function(model, block) == -1
        @test isnan(J.jumpy_value(model, block))
    end
    @test J.jumpy_free_set(set) == 0
    # Failed requests must leave the model usable.
    @test J.jumpy_value(model, x) ≈ 2.0
    @test J.jumpy_free_model(model) == 0
end

@testset "finite native callback dispatch" begin
    J = JuMPyHiGHS
    model = J.JuMPyDirect.model(J.Optimizer())
    x = only(J.JuMPyDirect.add_variables(model, 1))
    calls = Ref(0)
    callback = value -> begin
        calls[] += 1
        return value
    end
    values = (Int64(2), 2.0, x, 2x + 1, x^2, sin(x))
    for value in values
        before = calls[]
        @test J._dispatch_native(callback, value, J.NativeScalar) === value
        @test calls[] == before + 1
    end
    @test (@inferred J._dispatch_native(identity, x, J.NativeScalar)) === x
    @test (@inferred J._dispatch_native(
        +,
        Int64(2),
        3.0,
        J.NativeScalar,
        J.NativeScalar,
    )) === 5.0
    pair = (left, right) -> begin
        calls[] += 1
        return (left, right)
    end
    before = calls[]
    @test J._dispatch_native(pair, x, 2.0, J.NativeScalar, J.NativeScalar) === (x, 2.0)
    @test calls[] == before + 1
    for value in (Int32(2), Float32(2), 1 // 2, "unsupported", nothing)
        before = calls[]
        @test_throws ErrorException J._dispatch_native(callback, value, J.NativeScalar)
        @test_throws ErrorException J._dispatch_native(
            pair,
            value,
            x,
            J.NativeScalar,
            J.NativeScalar,
        )
        @test_throws ErrorException J._dispatch_native(
            pair,
            x,
            value,
            J.NativeScalar,
            J.NativeScalar,
        )
        @test calls[] == before
    end
end

@testset "native C ABI generated constraints" begin
    J = JuMPyHiGHS
    model = J.jumpy_new_model()
    @test model != C_NULL
    function apply_native(head::String, args...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_apply(
            model,
            Cstring(pointer(head)),
            pointer(argv),
            Clonglong(length(argv)),
        )
    end
    integer(value) = J.jumpy_integer_constant(model, Clonglong(value))
    block = J.jumpy_variables(model, Clonglong(3))
    indices = Clonglong[0, 1, 2]
    iterator = GC.@preserve indices J.jumpy_integer_iterator(
        model,
        pointer(indices),
        Clonglong(length(indices)),
    )
    demand = Cdouble[2.0, 6.0, 12.0]
    data = GC.@preserve demand J.jumpy_float_array(
        model,
        pointer(demand),
        Clonglong(length(demand)),
    )
    @test iterator != C_NULL
    @test data != C_NULL
    index = apply_native("+", iterator, integer(1))
    variables = apply_native("getindex", block, index)
    demands = apply_native("getindex", data, index)
    @test J._unbox(variables) isa J.GenOpt.ExprTemplate{J.NativeVariable}
    @test J._unbox(demands) isa J.NativeIterator
    # Integer iterator coefficients must be promoted to the model's Float64
    # coefficient type when expanding i * x[i] into affine rows.
    scaled = apply_native("*", index, variables)
    rows = apply_native("-", scaled, demands)
    @test J._unbox(rows) isa J.GenOpt.ExprTemplate{J.NativeAffine}
    set = J.jumpy_greater_than(0.0)
    @test J.jumpy_add_constraint(model, rows, set) == 0
    @test J.jumpy_free_set(set) == 0
    backend = JuMP.backend(J._get(model).model)
    @test MOI.get(
        backend,
        MOI.NumberOfConstraints{
            MOI.ScalarAffineFunction{Float64},
            MOI.GreaterThan{Float64},
        }(),
    ) == 3
    scalar_variables = [apply_native("getindex", block, integer(i)) for i = 1:3]
    objective = foldl(scalar_variables) do left, right
        apply_native("+", left, right)
    end
    @test J.jumpy_set_objective_sense(model, Cint(0)) == 0
    @test J.jumpy_set_objective_function(model, objective) == 0
    GC.gc(true)
    @test J.jumpy_optimize(model) == Int(MOI.OPTIMAL)
    @test [J.jumpy_value(model, x) for x in scalar_variables] ≈ [2.0, 3.0, 4.0]
    @test J.jumpy_value(model, objective) ≈ 9.0
    @test J.jumpy_free_model(model) == 0
end
