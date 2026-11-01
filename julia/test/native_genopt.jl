module NativeGenOptBenchmarkTests

using Test
import GenOpt
import HiGHS
import MathOptInterface as MOI
include("../../benchmarks/paper/native_genopt.jl")
const Experiment = NativeGenOptBenchmark
const F, S = Experiment.F, Experiment.S

function native_rows(backend, nr, nnz)
    rows, nonzeros = Ref{HiGHS.HighsInt}(0), Ref{HiGHS.HighsInt}(0)
    lower, upper = zeros(nr), zeros(nr)
    starts = Vector{HiGHS.HighsInt}(undef, nr + 1)
    indices, values = Vector{HiGHS.HighsInt}(undef, nnz), zeros(nnz)
    Experiment.Direct.check(
        HiGHS.Highs_getRowsByRange(
            backend,
            0,
            nr - 1,
            rows,
            lower,
            upper,
            nonzeros,
            starts,
            indices,
            values,
        ),
    )
    @test rows[] == nr
    starts[end] = nonzeros[]
    functions =
        [Dict(indices[k] => values[k] for k = (starts[row]+1):starts[row+1]) for row = 1:nr]
    return functions, lower, upper
end

@testset "native GenOpt paths match every LQCP row, bound, cost and solve" begin
    for n in (3, 5)
        data = Experiment.Direct.lqcp_data(n)
        reference, _ = Experiment.Direct.construct(n; solve = true)
        expected_objective = HiGHS.Highs_getObjectiveValue(reference)
        Experiment.dispose(reference)
        for mode in Experiment.MODES
            model, timings = Experiment.construct(n, mode; solve = true)
            try
                backend = model.backend
                @test HiGHS.Highs_getNumCol(backend) == data.nv
                @test HiGHS.Highs_getNumRow(backend) == data.nr
                functions, lower, upper = native_rows(backend, data.nr, length(data.values))
                @test lower == upper == data.rhs
                for row = 1:data.nr
                    expected = Dict(
                        data.columns[k] => data.values[k] for
                        k = (data.starts[row]+1):data.starts[row+1]
                    )
                    @test keys(functions[row]) == keys(expected)
                    @test all(
                        isapprox(value, expected[index]; atol = 1e-14) for
                        (index, value) in functions[row]
                    )
                end
                costs, col_lower, col_upper = zeros(data.nv), zeros(data.nv), zeros(data.nv)
                ncol, nnz = Ref{HiGHS.HighsInt}(0), Ref{HiGHS.HighsInt}(0)
                Experiment.Direct.check(
                    HiGHS.Highs_getColsByRange(
                        backend,
                        0,
                        data.nv - 1,
                        ncol,
                        costs,
                        col_lower,
                        col_upper,
                        nnz,
                        C_NULL,
                        C_NULL,
                        C_NULL,
                    ),
                )
                @test costs ≈ data.costs
                @test col_lower == data.lower
                @test col_upper == data.upper
                # HiGHS fills missing diagonal entries during the QP solve.
                @test HiGHS.Highs_getHessianNumNz(backend) == data.nv
                @test HiGHS.Highs_getModelStatus(backend) == HiGHS.kHighsModelStatusOptimal
                if startswith(mode, "grouped-")
                    @test MOI.get(backend, MOI.TerminationStatus()) == MOI.OPTIMAL
                    @test MOI.get(backend, MOI.ResultCount()) == 1
                    @test MOI.get(
                        backend,
                        MOI.NumberOfConstraints{
                            MOI.VariableIndex,
                            MOI.GreaterThan{Float64},
                        }(),
                    ) == data.nv
                    @test MOI.get(
                        backend,
                        MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}}(),
                    ) == data.nv
                end
                @test HiGHS.Highs_getObjectiveValue(backend) ≈ expected_objective atol =
                    1e-8
                @test timings.seconds ≈
                      timings.setup_seconds + timings.build_seconds + timings.run_seconds
                if backend isa HiGHS.Optimizer && mode != "grouped-stock"
                    @test length(backend.affine_constraint_info) == data.nr
                    @test MOI.is_valid(backend, Experiment.CI(data.nr))
                    @test MOI.get(backend, MOI.ConstraintSet(), Experiment.CI(data.nr)) ==
                          S(0.0)
                    @test MOI.get(
                        backend,
                        MOI.ConstraintFunction(),
                        Experiment.CI(data.nr),
                    ).constant == 0
                end
            finally
                Experiment.dispose(model.backend)
            end
        end
    end
end

@testset "native term columns merge duplicates and normalize constants" begin
    data = Experiment.Direct.lqcp_data(3)
    y = GenOpt.ContiguousArrayOfVariables(0, (data.nv,))
    i = GenOpt.IteratorIndex(1)
    expr = Experiment.snf(
        :+,
        Experiment.snf(:getindex, y, i),
        Experiment.snf(:*, -1, Experiment.snf(:getindex, y, i)),
        Experiment.snf(:*, 2, Experiment.snf(:getindex, y, 1)),
        Experiment.snf(:+, i, 3),
    )
    generator =
        GenOpt.FunctionGenerator{F}(expr, GenOpt.Iterator[GenOpt.Iterator([1, 2, 3])])
    context = GenOpt._BatchContext(generator.iterators)
    plan = GenOpt._batch_polynomial(Float64, generator.func, context)
    for columns in (false, true), mapped in (false, true)
        model = Experiment.setup(data, mapped)
        model.columns = columns
        try
            indices = Experiment.add_plan(model, plan, S(7.0), 3; batch_size = 2)
            @test [ci.value for ci in indices] == [1, 2, 3]
            @test model.calls == 2
            functions, lower, upper = native_rows(model.backend, 3, 9)
            @test all(f == Dict(0 => 2.0) for f in functions)
            @test lower == upper == [3.0, 2.0, 1.0]
            @test isempty(Experiment.add_plan(model, plan, S(7.0), 0))
            @test_throws ArgumentError Experiment.add_plan(
                model,
                plan,
                S(7.0),
                3;
                batch_size = 0,
            )
            @test_throws DimensionMismatch MOI.add_constraints(model, F[], [S(0.0)])
            invalid = F([MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(0))], 0.0)
            @test_throws MOI.InvalidIndex MOI.add_constraints(model, [invalid], [S(0.0)])
            @test_throws MOI.ScalarFunctionConstantNotZero MOI.add_constraints(
                model,
                [F(MOI.ScalarAffineTerm{Float64}[], 1.0)],
                [S(0.0)],
            )
            @test model.nr == 3
        finally
            Experiment.dispose(model.backend)
        end
    end
    starts, indices, values =
        zeros(Int32, 3), Int32[2, 1, 2, 0, 0, 0], [1.0, 4.0, -1.0, 1.0, 2.0, -3.0]
    Experiment.compact_rows!(starts, indices, values, 2, 3)
    @test starts == [0, 1, 1]
    @test indices == [1]
    @test values == [4.0]
    empty_model = Experiment.setup(data, false)
    try
        empty_model.columns = true
        constant_plan = GenOpt._batch_polynomial(Float64, Experiment.snf(:+, 3.0), context)
        Experiment.add_plan(empty_model, constant_plan, S(7.0), 3; batch_size = 2)
        functions, lower, upper = native_rows(empty_model.backend, 3, 0)
        @test all(isempty, functions)
        @test lower == upper == fill(4.0, 3)
    finally
        Experiment.dispose(empty_model.backend)
    end
end

@testset "bulk columns and bounds preserve HiGHS.jl variable metadata" begin
    backend = HiGHS.Optimizer()
    MOI.set(backend, MOI.Silent(), true)
    model = Experiment.NativeRows(backend, 0; bulk_variables = true, bulk_bounds = true)
    try
        first = MOI.add_variable(model)
        variables = MOI.add_variables(model, 2)
        @test first == MOI.VariableIndex(1)
        @test variables == MOI.VariableIndex.([2, 3])
        @test model.nv == HiGHS.Highs_getNumCol(backend) == 3
        @test HiGHS.column.(Ref(backend), variables) == [1, 2]
        lower = MOI.GreaterThan.([-2.0, 1.0])
        upper = MOI.LessThan.([3.0, 4.0])
        lower_indices = MOI.add_constraints(model, variables, lower)
        upper_indices = MOI.add_constraints(model, variables, upper)
        for i in eachindex(variables)
            @test MOI.is_valid(backend, variables[i])
            @test MOI.is_valid(backend, lower_indices[i])
            @test MOI.is_valid(backend, upper_indices[i])
            @test MOI.get(backend, MOI.ConstraintSet(), lower_indices[i]) == lower[i]
            @test MOI.get(backend, MOI.ConstraintSet(), upper_indices[i]) == upper[i]
        end
        @test isempty(MOI.add_variables(model, 0))
        @test isempty(
            MOI.add_constraints(model, MOI.VariableIndex[], MOI.GreaterThan{Float64}[]),
        )
        @test_throws ArgumentError MOI.add_variables(model, -1)
        @test_throws DimensionMismatch MOI.add_constraints(model, variables, lower[1:1])
        @test model.nv == 3
    finally
        Experiment.dispose(backend)
    end
end

@testset "native GenOpt runner" begin
    @test_throws ArgumentError Experiment.construct(3, "unknown")
    @test_throws ArgumentError Experiment.construct(2, "genopt-columns")
    redirect_stdout(devnull) do
        @test Experiment.main([3]; repeats = 1) === nothing
    end
end

end # module NativeGenOptBenchmarkTests
