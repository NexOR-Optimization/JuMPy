module MOIHiGHSBenchmarkTests

using Test
import HiGHS
import JSON
import MathOptInterface as MOI
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

@testset "zero-limit MOI benchmark and JSON samples" begin
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
    mktemp() do _, io
        redirect_stdout(io) do
            @test Benchmark.main([3]; repeats = 1, batch_size = 7) === nothing
        end
        flush(io)
        seekstart(io)
        records = JSON.parse.(filter(line -> startswith(line, "{"), readlines(io)))
        @test length(records) == 3
        @test records[1]["event"] == "environment"
        @test Set(keys(records[1]["native_bulk_methods"])) ==
              Set(["variables", "bounds", "rows"])
        @test Set(row["mode"] for row in records[2:3]) == Set(["bulk", "scalar"])
        @test all(
            row["variables"] == 19 && row["constraints"] == 16 for row in records[2:3]
        )
    end
end

end # module MOIHiGHSBenchmarkTests
