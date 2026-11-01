module PaperBenchmarkTests
using Test
import JuMP
import NexOR
import HiGHS
import MathOptInterface as MOI
include("../../benchmarks/paper/run_jump.jl")

@testset "paper benchmark runner stays local with NexOR" begin
    @test parse_args(["--backend", "nexor", "--sizes", "3,5"]) ==
          Dict("--backend" => "nexor", "--sizes" => "3,5")
    withenv("NEXOR_API_KEY" => "", "NEXOR_SERVER_URL" => "http://127.0.0.1:1") do
        mktempdir() do directory
            for mode in ("cached", "direct", "genopt")
                path = joinpath(directory, "$mode.mof.json")
                @test main([
                    "--backend",
                    "nexor",
                    "--model",
                    "facility",
                    "--mode",
                    mode,
                    "--sizes",
                    "3",
                    "--warmup",
                    "3",
                    "--repeats",
                    "1",
                    "--export",
                    path,
                ]) === nothing
                @test isfile(path)
            end
        end
    end
    @test main([
        "--backend",
        "highs",
        "--model",
        "lqcp",
        "--mode",
        "genopt",
        "--sizes",
        "3",
        "--warmup",
        "0",
        "--repeats",
        "1",
    ]) === nothing
    @test_throws ErrorException main(["--backend", "storage", "--model", "lqcp"])
end

@testset "grouped Julia paper formulations" begin
    for (factory, name, n) in (
        (NexOR.Optimizer, :lqcp, 3),
        (NexOR.Optimizer, :lqcp, 5),
        (NexOR.Optimizer, :facility, 3),
        (HiGHS.Optimizer, :lqcp, 3),
    )
        reference = JuMP.concrete_direct_model(factory())
        candidate = JuMP.concrete_direct_model(factory())
        JuMP.set_silent(reference)
        JuMP.set_silent(candidate)
        JuMP.set_string_names_on_creation(reference, false)
        getfield(PaperModels, name)(reference, n)
        getfield(GroupedPaperModels, name)(candidate, n)
        @test JuMP.num_variables(candidate) == JuMP.num_variables(reference)
        F = MOI.get(JuMP.backend(reference), MOI.ObjectiveFunctionType())
        @test MOI.get(JuMP.backend(candidate), MOI.ObjectiveFunction{F}()) ≈
              MOI.get(JuMP.backend(reference), MOI.ObjectiveFunction{F}())
        for (F, S) in MOI.get(JuMP.backend(reference), MOI.ListOfConstraintTypesPresent())
            expected = MOI.get(JuMP.backend(reference), MOI.ListOfConstraintIndices{F,S}())
            actual = MOI.get(JuMP.backend(candidate), MOI.ListOfConstraintIndices{F,S}())
            @test length(actual) == length(expected)
            remaining = copy(actual)
            for index in expected
                f = MOI.get(JuMP.backend(reference), MOI.ConstraintFunction(), index)
                s = MOI.get(JuMP.backend(reference), MOI.ConstraintSet(), index)
                match = findfirst(remaining) do other
                    return MOI.get(JuMP.backend(candidate), MOI.ConstraintSet(), other) == s && isapprox(
                        MOI.get(JuMP.backend(candidate), MOI.ConstraintFunction(), other),
                        f,
                    )
                end
                @test match !== nothing
                match === nothing || deleteat!(remaining, match)
            end
        end
        @test JuMP.termination_status(candidate) == MOI.OPTIMIZE_NOT_CALLED
    end
    model = JuMP.concrete_direct_model(NexOR.Optimizer())
    @test_throws ArgumentError GroupedPaperModels.lqcp(model, 2)
    @test_throws ArgumentError GroupedPaperModels.facility(model, 0)
end
end
