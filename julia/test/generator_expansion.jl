using Test
import GenOpt
import MathOptInterface as MOI

include("../../benchmarks/paper/generator_expansion.jl")

@testset "generator expansion benchmark families" begin
    for (kind, expected_variables, expected_rows) in ((:coordinate, 51, 48), (:pde, 16, 6))
        variables, generator = if kind == :coordinate
            GeneratorExpansionBenchmark.coordinate_family(3)
        else
            GeneratorExpansionBenchmark.pde_family(3)
        end
        model = MOI.Utilities.Model{Float64}()
        MOI.add_variables(model, variables)
        indices = GenOpt.add_constraints(model, generator, MOI.EqualTo(0.0))
        @test MOI.get(model, MOI.NumberOfVariables()) == expected_variables
        @test length(indices) == expected_rows
    end
end
