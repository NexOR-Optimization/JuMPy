module DirectHiGHSBenchmarkTests

using Test
import HiGHS
import JSON
import JuMP
include("../../benchmarks/paper/direct_highs.jl")
include("../../benchmarks/paper/models.jl")
const Direct = DirectHiGHSBenchmark

@testset "hardcoded LQCP arrays match the JuMP MOF fixture" begin
    fixture = JSON.parsefile(joinpath(@__DIR__, "../../tests/fixtures/lqcp3.mof.json"))
    data = Direct.lqcp_data(3)
    @test data.nv == length(fixture["variables"]) == 19
    @test data.nr == 16
    @test length(data.values) == 61
    variables = Dict(v["name"] => i for (i, v) in enumerate(fixture["variables"]))
    objective = fixture["objective"]["function"]
    costs, diagonal = zeros(data.nv), zeros(data.nv)
    for term in objective["affine_terms"]
        costs[variables[term["variable"]]] += term["coefficient"]
    end
    for term in objective["quadratic_terms"]
        @test term["variable_1"] == term["variable_2"]
        diagonal[variables[term["variable_1"]]] += term["coefficient"]
    end
    @test data.costs ≈ costs
    @test data.offset ≈ objective["constant"]
    actual_diagonal = zeros(data.nv)
    for col = 1:data.nv, k = (data.qstarts[col]+1):data.qstarts[col+1]
        @test data.qindices[k] == col - 1
        actual_diagonal[col] += data.qvalues[k]
    end
    @test actual_diagonal ≈ diagonal
    lower, upper = fill(-Inf, data.nv), fill(Inf, data.nv)
    expected_rows = Set()
    for constraint in fixture["constraints"]
        f, s = constraint["function"], constraint["set"]
        if f["type"] == "Variable"
            col = variables[f["name"]]
            if s["type"] == "GreaterThan"
                lower[col] = s["lower"]
            else
                upper[col] = s["upper"]
            end
        else
            @test s["type"] == "EqualTo" && s["value"] == 0
            @test f["constant"] == 0
            terms = [
                (variables[t["variable"]] - 1, round(t["coefficient"]; digits = 12)) for
                t in f["terms"]
            ]
            push!(expected_rows, Tuple(sort(terms)))
        end
    end
    actual_rows = Set()
    for row = 1:data.nr
        terms = [
            (Int(data.columns[k]), round(data.values[k]; digits = 12)) for
            k = (data.starts[row]+1):data.starts[row+1]
        ]
        push!(actual_rows, Tuple(sort(terms)))
    end
    @test length(actual_rows) == length(expected_rows) == data.nr
    @test actual_rows == expected_rows
    @test data.lower == lower
    @test data.upper == upper
    @test all(iszero, data.rhs)
    @test_throws ArgumentError Direct.lqcp_data(2)
    @test_throws ErrorException Direct.check(HiGHS.kHighsStatusError)
    @test Direct.check(HiGHS.kHighsStatusWarning) === nothing
end

@testset "hardcoded HiGHS model agrees with the original JuMP solve" begin
    for n in (3, 5)
        reference = JuMP.concrete_direct_model(HiGHS.Optimizer())
        JuMP.set_silent(reference)
        JuMP.set_attribute(reference, "threads", 1)
        JuMP.set_attribute(reference, "presolve", "off")
        PaperModels.lqcp(reference, n)
        JuMP.optimize!(reference)
        highs, timings = Direct.construct(n; solve = true)
        try
            @test HiGHS.Highs_getModelStatus(highs) == HiGHS.kHighsModelStatusOptimal
            @test HiGHS.Highs_getNumCol(highs) == (n + 1)^2 + n
            @test HiGHS.Highs_getNumRow(highs) == (n + 1)^2
            @test HiGHS.Highs_getObjectiveValue(highs) ≈ JuMP.objective_value(reference) atol =
                1e-8
            @test timings.seconds ≈
                  timings.assembly_seconds + timings.load_seconds + timings.run_seconds
        finally
            HiGHS.Highs_destroy(highs)
        end
    end
    highs, _ = Direct.construct(3)
    try
        @test HiGHS.Highs_getModelStatus(highs) == HiGHS.kHighsModelStatusTimeLimit
    finally
        HiGHS.Highs_destroy(highs)
    end
    redirect_stdout(devnull) do
        @test Direct.main([3]; repeats = 1) === nothing
    end
end

end # module DirectHiGHSBenchmarkTests
