module PaperBenchmarkTests
using Test
import JuMP
import MathOptInterface as MOI
include("../../benchmarks/paper/storage.jl")
include("../../benchmarks/paper/models.jl")
include("../../benchmarks/paper/library_entry.jl")
module RunnerTests
include("../../benchmarks/paper/run_jump.jl")
using Test
@test parse_args(["--model", "facility", "--sizes", "3,5"]) ==
      Dict("--model" => "facility", "--sizes" => "3,5")
@test main([
    "--model",
    "facility",
    "--sizes",
    "3",
    "--warmup",
    "3",
    "--repeats",
    "1",
    "--mode",
    "direct",
]) === nothing
@test main([
    "--model",
    "lqcp",
    "--sizes",
    "3",
    "--warmup",
    "3",
    "--repeats",
    "1",
    "--mode",
    "cached",
]) === nothing
mktempdir() do directory
    path = joinpath(directory, "model.mof.json")
    @test main([
        "--model",
        "lqcp",
        "--sizes",
        "3",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--export",
        path,
    ]) === nothing
    @test isfile(path)
end
end
module HighsBackend
withenv("JUMPY_PAPER_BACKEND" => "highs") do
    include("../../benchmarks/paper/library_entry.jl")
end
end

@testset "paper reference dimensions" begin
    for n in (3, 5)
        model = JuMP.direct_model(PaperStorage.Model{Float64}())
        PaperModels.facility(model, n)
        count = n * (n + 1)^2
        @test JuMP.num_variables(model) == 2n + 4count + 1
        @test MOI.get(
            JuMP.backend(model),
            MOI.NumberOfConstraints{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.LessThan{Float64},
            }(),
        ) == count
        model = JuMP.direct_model(PaperStorage.Model{Float64}())
        PaperModels.lqcp(model, n)
        @test JuMP.num_variables(model) == (n + 1)^2 + n
        @test MOI.get(
            JuMP.backend(model),
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.EqualTo{Float64},
            }(),
        ) == (n + 1)^2
    end
    model = JuMP.direct_model(PaperStorage.Model{Float64}())
    PaperModels.latency(model)
    @test JuMP.num_variables(model) == 2
end

@testset "native block bounds and integrality" begin
    J = JuMPyPaper
    for (set, constructor) in (
        (MOI.GreaterThan(0.0), () -> J.jumpy_greater_than(0.0)),
        (MOI.LessThan(1.0), () -> J.jumpy_less_than(1.0)),
        (MOI.ZeroOne(), J.jumpy_zero_one),
        (MOI.Integer(), J.jumpy_integer),
    )
        model = J.jumpy_new_model()
        block = J.jumpy_variables(model, Clonglong(3))
        id = constructor()
        try
            @test J.jumpy_add_variable_constraints(model, block, id) == 0
            backend = JuMP.backend(J._get(model).model)
            @test MOI.get(
                backend,
                MOI.NumberOfConstraints{MOI.VariableIndex,typeof(set)}(),
            ) == 3
            for variable in J._unbox(block)
                index = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(
                    JuMP.index(variable).value,
                )
                @test MOI.get(backend, MOI.ConstraintSet(), index) == set
            end
        finally
            J.jumpy_free_set(id)
            J.jumpy_free_model(model)
        end
    end
    first = J.jumpy_new_model()
    second = J.jumpy_new_model()
    foreign = J.jumpy_variables(second, Clonglong(1))
    id = J.jumpy_greater_than(0.0)
    try
        redirect_stderr(devnull) do
            @test J.jumpy_add_variable_constraints(first, foreign, id) == -1
        end
    finally
        J.jumpy_free_set(id)
        J.jumpy_free_model(first)
        J.jumpy_free_model(second)
    end
end

@testset "bulk bounds update cached models and invalidate results" begin
    J = JuMPyPaper.JuMPyModel
    model = J.model(HighsBackend.JuMPyPaper.Optimizer)
    JuMP.set_silent(model)
    variables = J.add_variables(model, 2)
    J.add_variable_constraints(model, variables, MOI.LessThan(2.0))
    JuMP.@objective(model, Max, sum(variables))
    JuMP.optimize!(model)
    @test JuMP.objective_value(model) ≈ 4.0
    J.add_variable_constraints(model, variables, MOI.ZeroOne())
    @test all(JuMP.is_binary, variables)
    @test JuMP.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ 2.0
end

@testset "benchmark diagnostics support storage and solvers" begin
    for J in (JuMPyPaper, HighsBackend.JuMPyPaper)
        model = J.jumpy_new_model()
        block = J.jumpy_variables(model, Clonglong(2))
        id = J.jumpy_greater_than(0.0)
        try
            @test J.jumpy_add_variable_constraints(model, block, id) == 0
            @test J.jumpy_paper_num_variables(model) == 2
            @test MOI.get(JuMP.backend(J._get(model).model), MOI.Silent())
            mktempdir() do directory
                path = joinpath(directory, "model.mof.json")
                GC.@preserve path begin
                    @test J.jumpy_paper_write_model(model, Cstring(pointer(path))) == 0
                end
                @test isfile(path)
            end
            @test J.jumpy_paper_gc() === nothing
        finally
            J.jumpy_free_set(id)
            J.jumpy_free_model(model)
        end
    end
end
end
