module MatrixHiGHSBenchmarkTests

using Test
import HiGHS
import MathOptInterface as MOI
include("../../benchmarks/paper/matrix_highs.jl")
const Experiment = MatrixHiGHSBenchmark
const F, S, CI = Experiment.F, Experiment.S, Experiment.CI

@testset "transposed CSC wrapping shares coefficients and preserves indices" begin
    data = Experiment.Direct.lqcp_data(3)
    model = Experiment.matrix_rows(data)
    at = parent(model.coefficients)
    @test size(model.coefficients) == (data.nr, data.nv)
    @test at.nzval === data.values
    @test at.colptr == data.starts .+ 1
    @test at.rowval == data.columns .+ 1
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == data.nr
    functions, sets = Experiment.row_functions(data, 1:data.nr)
    for row = 1:data.nr
        @test MOI.is_valid(model, CI(row))
        @test MOI.get(model, MOI.ConstraintFunction(), CI(row)) ≈ functions[row]
        @test MOI.get(model, MOI.ConstraintSet(), CI(row)) == sets[row]
    end
    before = (copy(at.colptr), copy(at.rowval), copy(at.nzval))
    highs = Experiment.create_highs()
    try
        Experiment.copy_to(highs, model, data)
        @test HiGHS.Highs_getNumRow(highs) == data.nr
        @test (at.colptr, at.rowval, at.nzval) == before
        wrong = Experiment.empty_rows(data.nv - 1)
        @test_throws DimensionMismatch Experiment.copy_to(highs, wrong, data)
        @test !wrong.final_touch
    finally
        HiGHS.Highs_destroy(highs)
    end
end

@testset "eager row batches preserve ownership, normalization and row numbers" begin
    data = Experiment.Direct.lqcp_data(5)
    model = Experiment.empty_rows(data.nv)
    @test isempty(MOI.add_constraints(model, F[], S[]))
    for first_row = 1:7:data.nr
        rows = first_row:min(first_row+6, data.nr)
        functions, sets = Experiment.row_functions(data, rows)
        indices = MOI.add_constraints(model, functions, sets)
        @test [ci.value for ci in indices] == collect(rows)
        for (i, ci) in enumerate(indices)
            @test MOI.get(model, MOI.ConstraintFunction(), ci) ≈ functions[i]
            @test MOI.get(model, MOI.ConstraintSet(), ci) == sets[i]
        end
        # Storage owns values; modifying the submitted functions cannot alter it.
        old = MOI.get(model, MOI.ConstraintFunction(), first(indices))
        empty!(functions[1].terms)
        @test MOI.get(model, MOI.ConstraintFunction(), first(indices)) ≈ old
    end
    wrapped = Experiment.matrix_rows(data)
    @test model.coefficients == wrapped.coefficients
    @test model.constants == wrapped.constants
    @test model.sets.set_ids == wrapped.sets.set_ids
    x = MOI.VariableIndex(1)
    noncanonical = F(
        [
            MOI.ScalarAffineTerm(2.0, MOI.VariableIndex(2)),
            MOI.ScalarAffineTerm(1.0, x),
            MOI.ScalarAffineTerm(3.0, x),
        ],
        0.0,
    )
    index = only(MOI.add_constraints(model, [noncanonical], [S(7.0)]))
    @test MOI.get(model, MOI.ConstraintFunction(), index) ≈
          MOI.Utilities.canonical(noncanonical)
    @test MOI.get(model, MOI.ConstraintSet(), index) == S(7.0)
    before = copy(parent(model.coefficients).nzval)
    @test_throws DimensionMismatch MOI.add_constraints(model, [noncanonical], S[])
    invalid = F([MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(data.nv + 1))], 0.0)
    @test_throws MOI.InvalidIndex MOI.add_constraints(model, [invalid], [S(0.0)])
    @test_throws MOI.ScalarFunctionConstantNotZero MOI.add_constraints(
        model,
        [F(MOI.ScalarAffineTerm{Float64}[], 1.0)],
        [S(0.0)],
    )
    @test parent(model.coefficients).nzval == before
    Experiment.MOI.Utilities.final_touch(model, nothing)
    @test_throws MOI.AddConstraintNotAllowed MOI.add_constraints(
        model,
        [noncanonical],
        [S(0.0)],
    )
    @test_throws MOI.InvalidIndex MOI.get(model, MOI.ConstraintFunction(), CI(0))
    @test_throws MOI.InvalidIndex MOI.get(model, MOI.ConstraintSet(), CI(0))
end

function check_native_rows(highs, data)
    nr, nnz = Ref{HiGHS.HighsInt}(0), Ref{HiGHS.HighsInt}(0)
    lower, upper = zeros(data.nr), zeros(data.nr)
    starts = Vector{HiGHS.HighsInt}(undef, data.nr + 1)
    columns = similar(data.columns)
    values = similar(data.values)
    Experiment.Direct.check(
        HiGHS.Highs_getRowsByRange(
            highs,
            0,
            data.nr - 1,
            nr,
            lower,
            upper,
            nnz,
            starts,
            columns,
            values,
        ),
    )
    @test nr[] == data.nr
    @test nnz[] == length(data.values)
    starts[end] = nnz[]
    @test lower == upper == data.rhs
    for row = 1:data.nr
        actual = Dict(columns[k] => values[k] for k = (starts[row]+1):starts[row+1])
        expected = Dict(
            data.columns[k] => data.values[k] for
            k = (data.starts[row]+1):data.starts[row+1]
        )
        @test actual == expected
    end
end

@testset "all insertion paths load the same native coefficients and solve" begin
    for n in (3, 5)
        data = Experiment.Direct.lqcp_data(n)
        reference, _ = Experiment.Direct.construct(n; solve = true)
        expected_objective = HiGHS.Highs_getObjectiveValue(reference)
        HiGHS.Highs_destroy(reference)
        for mode in Experiment.MODES
            highs, timings = Experiment.construct(n, mode; solve = true, batch_size = 7)
            try
                check_native_rows(highs, data)
                @test HiGHS.Highs_getNumCol(highs) == data.nv
                @test HiGHS.Highs_getModelStatus(highs) == HiGHS.kHighsModelStatusOptimal
                @test HiGHS.Highs_getObjectiveValue(highs) ≈ expected_objective atol = 1e-8
                @test timings.seconds > 0
            finally
                HiGHS.Highs_destroy(highs)
            end
        end
    end
    @test_throws ArgumentError Experiment.construct(3, "unknown")
    @test_throws ArgumentError Experiment.construct(3, "direct"; batch_size = 0)
    redirect_stdout(devnull) do
        @test Experiment.main([3]; repeats = 1, batch_size = 7) === nothing
    end
end

end # module MatrixHiGHSBenchmarkTests
