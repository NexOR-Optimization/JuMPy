module MOIHiGHSBenchmarkTests

using Test
import HiGHS
import MathOptInterface as MOI
using DataFrames: DataFrame, nrow
include("../../benchmarks/paper/moi_highs.jl")
include("../../benchmarks/paper/direct_highs.jl")
const Benchmark = MOIHiGHSBenchmark
const Direct = DirectHiGHSBenchmark
const F = MOI.ScalarAffineFunction{Float64}
const E = MOI.EqualTo{Float64}

@testset "public MOI LQCP matches the hardcoded model" begin
    for n in (3, 5), bulk in (true, false), batch_size in (1, 7, 65_536)
        data = Direct.lqcp_data(n)
        reference, _ = Direct.construct(n; solve = true)
        expected_objective = HiGHS.Highs_getObjectiveValue(reference)
        HiGHS.Highs_destroy(reference)
        model, timings = Benchmark.construct(n; bulk, batch_size, solve = true)
        try
            @test model isa HiGHS.Optimizer
            variables = MOI.get(model, MOI.ListOfVariableIndices())
            @test length(variables) == data.nv
            @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE
            @test MOI.get(
                model,
                MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
            ) == data.nv
            @test MOI.get(
                model,
                MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}}(),
            ) == data.nv
            for variable in variables
                lower = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(
                    variable.value,
                )
                upper = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(
                    variable.value,
                )
                @test MOI.get(model, MOI.ConstraintSet(), lower).lower ==
                      data.lower[variable.value]
                @test MOI.get(model, MOI.ConstraintSet(), upper).upper ==
                      data.upper[variable.value]
                @test MOI.get(model, MOI.VariableName(), variable) == ""
            end
            constraints = MOI.get(model, MOI.ListOfConstraintIndices{F,E}())
            @test length(constraints) == data.nr
            for (row, index) in enumerate(constraints)
                f = MOI.get(model, MOI.ConstraintFunction(), index)
                expected = Dict(
                    data.columns[k]+1 => data.values[k] for
                    k = (data.starts[row]+1):data.starts[row+1]
                )
                actual = Dict(term.variable.value => term.coefficient for term in f.terms)
                @test keys(actual) == keys(expected)
                @test all(
                    isapprox(value, expected[column]; atol = 1e-14) for
                    (column, value) in actual
                )
                @test iszero(f.constant)
                @test MOI.get(model, MOI.ConstraintSet(), index) == E(data.rhs[row])
            end
            objective = MOI.get(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            )
            affine, diagonal = zeros(data.nv), zeros(data.nv)
            for term in objective.affine_terms
                affine[term.variable.value] += term.coefficient
            end
            for term in objective.quadratic_terms
                @test term.variable_1 == term.variable_2
                diagonal[term.variable_1.value] += term.coefficient
            end
            expected_diagonal = zeros(data.nv)
            for col = 1:data.nv, k = (data.qstarts[col]+1):data.qstarts[col+1]
                expected_diagonal[col] += data.qvalues[k]
            end
            @test affine ≈ data.costs
            @test diagonal ≈ expected_diagonal
            @test objective.constant ≈ data.offset
            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ expected_objective atol = 1e-8
            @test MOI.get(model, MOI.ResultCount()) == 1
            @test timings.seconds ≈
                  timings.setup_seconds + timings.construction_seconds + timings.run_seconds
            @test timings.variables_seconds +
                  timings.bounds_seconds +
                  timings.objective_seconds +
                  timings.rows_seconds <= timings.construction_seconds
        finally
            finalize(model)
        end
    end
end

@testset "zero-limit MOI benchmark and Markdown summary" begin
    @test_throws ArgumentError Benchmark.construct(2)
    @test_throws ArgumentError Benchmark.construct(3; batch_size = 0)
    model, _ = Benchmark.construct(3)
    try
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    finally
        finalize(model)
    end
    methods = Benchmark.bulk_methods()
    @test all(value isa Bool for value in values(methods))
    io = IOBuffer()
    result = Benchmark.main([3, 5]; repeats = 2, batch_size = 7, io)
    @test result.samples isa DataFrame
    @test result.summary isa DataFrame
    @test nrow(result.samples) == 8
    @test nrow(result.summary) == 4
    @test all(result.summary.runs .== 2)
    @test all(
        result.samples.variables .== (result.samples.size .+ 1) .^ 2 .+ result.samples.size,
    )
    @test all(result.samples.constraints .== (result.samples.size .+ 1) .^ 2)
    for row in eachrow(result.summary)
        @test row.min_seconds <= row.seconds <= row.max_seconds
    end
    markdown = String(take!(io))
    @test startswith(markdown, "|")
    @test occursin("Bulk", markdown)
    @test occursin("Runs", markdown)
    @test occursin("Total (ms)", markdown)
    @test occursin("true", markdown)
    @test occursin("false", markdown)
    @test !occursin("{", markdown)
    @test !occursin("Float64", markdown)
end

@testset "DataFrames combines repeated samples without mixing configurations" begin
    samples = DataFrame(
        size = [3, 3, 3, 5, 3],
        bulk = [true, true, false, true, true],
        batch_size = [7, 7, 7, 7, 1],
        variables = [19, 19, 19, 41, 19],
        constraints = [16, 16, 16, 36, 16],
        seconds = [1.0, 3.0, 9.0, 7.0, 5.0],
        rows_seconds = [0.2, 0.6, 0.8, 0.4, 0.3],
    )
    original = copy(samples)
    summary = Benchmark.summarize(samples)
    @test samples == original
    @test nrow(summary) == 4
    bulk = only(
        eachrow(
            summary[(summary.size .== 3) .& summary.bulk .& (summary.batch_size .== 7), :],
        ),
    )
    @test bulk.runs == 2
    @test bulk.seconds == 2.0
    @test bulk.rows_seconds ≈ 0.4
    @test bulk.min_seconds == 1.0
    @test bulk.max_seconds == 3.0
    scalar = only(eachrow(summary[.!summary.bulk, :]))
    @test scalar.runs == 1
    @test scalar.seconds == 9.0
    @test Benchmark.milliseconds([0.000123, 0.002345]) == [0.1, 2.3]
end

end # module MOIHiGHSBenchmarkTests
