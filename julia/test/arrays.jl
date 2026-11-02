# Variable arrays, vector constraints, data arrays and nonlinear operators of
# the C ABI, with a backend over a mock optimizer that stores the model.
using Test
import JuMPy
import HiGHS

module JuMPyMock
import JuMPy
import MathOptInterface
const _Mock = MathOptInterface.Utilities.MockOptimizer{
    MathOptInterface.Utilities.Model{Float64},
    Float64,
}
# A backend constructs its optimizer without arguments.
(::Type{_Mock})() =
    MathOptInterface.Utilities.MockOptimizer(MathOptInterface.Utilities.Model{Float64}())
JuMPy.@entrypoints(_Mock, (MOI.GreaterThan{Float64}, MOI.Nonnegatives, MOI.Zeros))
end

# A set with a value parameter, as `MathOptVRP.TimeWindows{W}`.
struct ParametricSet{W,T} <: JuMPy.MOI.AbstractScalarSet
    value::T
end

@testset "registry" begin
    MOI = JuMPy.MOI
    @test JuMPy.supported_sets(HiGHS.Optimizer) == (
        MOI.EqualTo{Float64},
        MOI.GreaterThan{Float64},
        MOI.Integer,
        MOI.Interval{Float64},
        MOI.LessThan{Float64},
        MOI.Semicontinuous{Float64},
        MOI.Semiinteger{Float64},
        MOI.ZeroOne,
    )
    @test JuMPy.set_name(MOI.LessThan{Float64}) == "MathOptInterface.LessThan"
    @test JuMPy.set_name(ParametricSet{:start,Float64}) == "Main.ParametricSet{start}"
    @test JuMPy.constructible(ParametricSet{:start,Float64})
    @test !JuMPy.constructible(ParametricSet{:start,String})
    @test !JuMPy.constructible(MOI.LessThan)
    @test JuMPy.registry((MOI.Zeros, MOI.Integer)) ==
          "set\t1\tMathOptInterface.Zeros\tdimension\tInt64\n" *
          "set\t2\tMathOptInterface.Integer\t\t\n"
    @test isempty(JuMPy.operators((MOI.Zeros,)))
end

@testset "vector functions and sets" begin
    J = JuMPyMock
    C = JuMPy.Client
    MOI = JuMPy.MOI
    m = J.jumpy_new_model()
    set = C.new_set(J, "MathOptInterface.Nonnegatives", 3)
    x = C.constrained_variables(J, m, (3,), set)
    @test J._unbox_value(x) isa Vector{J.NativeVariable}
    first = C.index(J, m, x, 1)
    @test J._unbox_value(first) === J._unbox_value(x)[1]
    tail = C.index(J, m, x, 2:3)
    @test J._unbox_value(tail) == J._unbox_value(x)[2:3]
    vector = C.concatenate(J, m, C.constant(J, m, 1.0), tail, first)
    @test J._unbox_value(vector) isa Vector{J.NativeAffine}
    @test length(J._unbox_value(vector)) == 4
    @test C.add_constraint(J, m, vector, "MathOptInterface.Zeros", 4) != C_NULL
    @test C.add_constraint(J, m, first, "MathOptInterface.GreaterThan", 1.0) != C_NULL
    inner = J.JuMP.backend(J._get(m).model)
    F, S = MOI.VectorAffineFunction{Float64}, MOI.Zeros
    @test MOI.get(inner, MOI.NumberOfConstraints{F,S}()) == 1
    F, S = MOI.VectorOfVariables, MOI.Nonnegatives
    @test MOI.get(inner, MOI.NumberOfConstraints{F,S}()) == 1
    redirect_stderr(devnull) do
        # Vector functions in scalar sets, and conversely.
        @test C.add_constraint(J, m, vector, "MathOptInterface.GreaterThan", 0.0) == C_NULL
        @test C.add_constraint(J, m, first, "MathOptInterface.Zeros", 1) == C_NULL
        @test C.index(J, m, x, 1, 1) == C_NULL
        @test C.constrained_variables(J, m, (1, 2, 3), set) == C_NULL
        @test C.concatenate(J, m, C.nonlinear(J, m, "sin", first)) == C_NULL
    end
    data = C.array(J, m, [1 2; 3 4])
    @test J._unbox_value(data) == [1 2; 3 4]
    @test J._unbox_value(C.index(J, m, data, 2, 1:2)) == [3, 4]
    @test J._unbox_value(C.array(J, m, [1.5, 2.5])) == [1.5, 2.5]
    expression = J._unbox_value(C.nonlinear(J, m, "sum_distances", data, vector))
    @test expression isa J.NativeNonlinear
    @test expression.head == :sum_distances
    @test expression.args == Any[[1 2; 3 4], J._unbox_value(vector)]
    @test J.jumpy_free_set(set) == 0
    @test J.jumpy_free_model(m) == 0
end
