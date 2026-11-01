# Bottom-up LQCP storage experiment. Only scalar equality rows are supported.
module MatrixHiGHSBenchmark

import HiGHS
import JSON
import MathOptInterface as MOI
using LinearAlgebra: Transpose, transpose
using SparseArrays: SparseMatrixCSC
include("direct_highs.jl")
const Direct = DirectHiGHSBenchmark
const H = HiGHS.HighsInt
const F = MOI.ScalarAffineFunction{Float64}
const S = MOI.EqualTo{Float64}
const CI = MOI.ConstraintIndex{F,S}

MOI.Utilities.@mix_of_scalar_sets(EqualityRows, MOI.EqualTo{T})
const CSR = Transpose{Float64,SparseMatrixCSC{Float64,H}}
const Rows = MOI.Utilities.MatrixOfConstraints{
    Float64,
    CSR,
    MOI.Utilities.Hyperrectangle{Float64},
    EqualityRows{Float64},
}

function matrix_rows(data)
    # CSC(A') is CSR(A). Its coefficients alias the direct builder's values.
    at = SparseMatrixCSC{Float64,H}(
        data.nv,
        data.nr,
        data.starts .+ H(1),
        data.columns .+ H(1),
        data.values,
    )
    sets = EqualityRows{Float64}()
    fill!(resize!(sets.set_ids, data.nr), 1)
    model = MOI.Utilities.MatrixOfConstraints{Float64}(
        transpose(at),
        MOI.Utilities.Hyperrectangle(data.rhs, data.rhs),
        sets,
    )
    MOI.Utilities.final_touch(model, nothing)
    return model
end

function empty_rows(nv)
    at = SparseMatrixCSC{Float64,H}(nv, 0, H[1], H[], Float64[])
    return MOI.Utilities.MatrixOfConstraints{Float64}(
        transpose(at),
        MOI.Utilities.Hyperrectangle{Float64}(),
        EqualityRows{Float64}(),
    )
end

function MOI.Utilities.final_touch(model::Rows, ::Nothing)
    empty!(model.caches)
    empty!(model.are_indices_mapped)
    model.final_touch = true
    return
end

function canonical_rows(functions::Vector{F}, sets::Vector{S}, nv)
    length(functions) == length(sets) || throw(DimensionMismatch("Functions and sets"))
    canonical = map(functions) do f
        MOI.throw_if_scalar_and_constant_not_zero(f, S)
        for term in f.terms
            1 <= term.variable.value <= nv || throw(MOI.InvalidIndex(term.variable))
        end
        return MOI.Utilities.is_canonical(f) ? f : MOI.Utilities.canonical(f)
    end
    return canonical
end

function MOI.add_constraints(model::Rows, functions::Vector{F}, sets::Vector{S})
    model.final_touch && throw(MOI.AddConstraintNotAllowed{F,S}("Matrix is finalized"))
    at = parent(model.coefficients)
    functions = canonical_rows(functions, sets, at.m)
    count = length(functions)
    old_rows, old_nnz = at.n, length(at.nzval)
    nnz = old_nnz + sum(f -> length(f.terms), functions; init = 0)
    resize!(at.colptr, old_rows + count + 1)
    resize!(at.rowval, nnz)
    resize!(at.nzval, nnz)
    resize!(model.constants.lower, old_rows + count)
    resize!(model.constants.upper, old_rows + count)
    resize!(model.sets.set_ids, old_rows + count)
    indices = Vector{CI}(undef, count)
    position = old_nnz + 1
    @inbounds for i in eachindex(functions)
        row = old_rows + i
        for term in functions[i].terms
            at.rowval[position] = term.variable.value
            at.nzval[position] = term.coefficient
            position += 1
        end
        at.colptr[row+1] = position
        model.constants.lower[row] = model.constants.upper[row] = sets[i].value
        model.sets.set_ids[row] = 1
        indices[i] = CI(row)
    end
    # Only the shape changes; the three backing vectors stay owned by the model.
    model.coefficients = transpose(
        SparseMatrixCSC{Float64,H}(at.m, old_rows + count, at.colptr, at.rowval, at.nzval),
    )
    return indices
end

function MOI.get(model::Rows, ::MOI.ConstraintFunction, ci::CI)
    MOI.is_valid(model, ci) || throw(MOI.InvalidIndex(ci))
    return MOI.Utilities.extract_function(model.coefficients, ci.value, 0.0)
end

function MOI.get(model::Rows, ::MOI.ConstraintSet, ci::CI)
    MOI.is_valid(model, ci) || throw(MOI.InvalidIndex(ci))
    return S(model.constants.lower[ci.value])
end

function create_highs(; solve = false)
    highs = HiGHS.Highs_create()
    highs == C_NULL && error("Could not create HiGHS")
    try
        Direct.check(HiGHS.Highs_setBoolOptionValue(highs, "output_flag", false))
        Direct.check(HiGHS.Highs_setIntOptionValue(highs, "threads", 1))
        Direct.check(HiGHS.Highs_setStringOptionValue(highs, "presolve", "off"))
        if !solve
            Direct.check(HiGHS.Highs_setDoubleOptionValue(highs, "time_limit", 0.0))
        end
    catch
        HiGHS.Highs_destroy(highs)
        rethrow()
    end
    return highs
end

function pass_model(highs, data, nr, starts, columns, values, lower, upper)
    Direct.check(
        HiGHS.Highs_passModel(
            highs,
            data.nv,
            nr,
            length(values),
            length(data.qvalues),
            HiGHS.kHighsMatrixFormatRowwise,
            HiGHS.kHighsHessianFormatTriangular,
            HiGHS.kHighsObjSenseMinimize,
            data.offset,
            data.costs,
            data.lower,
            data.upper,
            lower,
            upper,
            starts,
            columns,
            values,
            data.qstarts,
            data.qindices,
            data.qvalues,
            C_NULL,
        ),
    )
    return
end

function copy_to(highs, model::Rows, data)
    # Experimental matrix loader, not a complete MOI optimizer copy_to method.
    at = parent(model.coefficients)
    at.m == data.nv || throw(DimensionMismatch("Matrix and variable bounds"))
    MOI.Utilities.final_touch(model, nothing)
    # Preserve a valid, readable source matrix: only index vectors are copied.
    starts, columns = at.colptr .- H(1), at.rowval .- H(1)
    pass_model(
        highs,
        data,
        at.n,
        starts,
        columns,
        at.nzval,
        model.constants.lower,
        model.constants.upper,
    )
    return
end

function row_functions(data, rows)
    functions = Vector{F}(undef, length(rows))
    sets = Vector{S}(undef, length(rows))
    for (i, row) in enumerate(rows)
        entries = (data.starts[row]+1):data.starts[row+1]
        terms = Vector{MOI.ScalarAffineTerm{Float64}}(undef, length(entries))
        for (j, k) in enumerate(entries)
            terms[j] =
                MOI.ScalarAffineTerm(data.values[k], MOI.VariableIndex(data.columns[k]+1))
        end
        functions[i] = F(terms, 0.0)
        sets[i] = S(data.rhs[row])
    end
    return functions, sets
end

function add_native_functions(highs, functions::Vector{F}, sets::Vector{S}, nv)
    functions = canonical_rows(functions, sets, nv)
    starts = Vector{H}(undef, length(functions) + 1)
    nnz = sum(f -> length(f.terms), functions; init = 0)
    columns, values = Vector{H}(undef, nnz), Vector{Float64}(undef, nnz)
    rhs = Vector{Float64}(undef, length(sets))
    position = 1
    for i in eachindex(functions)
        starts[i] = position - 1
        for term in functions[i].terms
            columns[position] = term.variable.value - 1
            values[position] = term.coefficient
            position += 1
        end
        rhs[i] = sets[i].value
    end
    starts[end] = position - 1
    Direct.check(
        HiGHS.Highs_addRows(
            highs,
            length(functions),
            rhs,
            rhs,
            nnz,
            starts,
            columns,
            values,
        ),
    )
    return
end

function add_native_arrays(highs, data, rows)
    first_nz = Int(data.starts[first(rows)]) + 1
    last_nz = Int(data.starts[last(rows)+1])
    starts = data.starts[first(rows):(last(rows)+1)] .- data.starts[first(rows)]
    rhs = view(data.rhs, rows)
    Direct.check(
        HiGHS.Highs_addRows(
            highs,
            length(rows),
            rhs,
            rhs,
            last_nz - first_nz + 1,
            starts,
            view(data.columns, first_nz:last_nz),
            view(data.values, first_nz:last_nz),
        ),
    )
    return
end

const MODES = (
    "direct",
    "matrix-wrap",
    "matrix-batches",
    "native-one",
    "native-batches",
    "native-functions",
)

function construct(n, mode; solve = false, batch_size = 65_536)
    mode in MODES || throw(ArgumentError("Unknown mode: $mode"))
    batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
    if mode == "direct"
        return Direct.construct(n; solve)
    end
    started = time_ns()
    data = Direct.lqcp_data(n)
    assembled = time_ns()
    highs = create_highs(; solve)
    row_build, insertion, loading = UInt64(0), UInt64(0), UInt64(0)
    try
        if mode == "matrix-wrap"
            t = time_ns()
            model = matrix_rows(data)
            insertion += time_ns() - t
            t = time_ns()
            copy_to(highs, model, data)
            loading += time_ns() - t
        else
            model = mode == "matrix-batches" ? empty_rows(data.nv) : nothing
            if model === nothing
                t = time_ns()
                pass_model(highs, data, 0, H[0], H[], Float64[], Float64[], Float64[])
                loading += time_ns() - t
            end
            step = mode == "native-one" ? data.nr : batch_size
            for first_row = 1:step:data.nr
                rows = first_row:min(first_row+step-1, data.nr)
                if mode in ("matrix-batches", "native-functions")
                    t = time_ns()
                    functions, sets = row_functions(data, rows)
                    row_build += time_ns() - t
                    t = time_ns()
                    if model === nothing
                        add_native_functions(highs, functions, sets, data.nv)
                    else
                        MOI.add_constraints(model, functions, sets)
                    end
                    insertion += time_ns() - t
                else
                    t = time_ns()
                    add_native_arrays(highs, data, rows)
                    insertion += time_ns() - t
                end
            end
            if model !== nothing
                t = time_ns()
                copy_to(highs, model, data)
                loading += time_ns() - t
            end
        end
        loaded = time_ns()
        Direct.check(HiGHS.Highs_run(highs))
        finished = time_ns()
        return highs,
        (;
            seconds = (finished - started) / 1e9,
            assembly_seconds = (assembled - started) / 1e9,
            row_build_seconds = row_build / 1e9,
            insert_seconds = insertion / 1e9,
            load_seconds = loading / 1e9,
            run_seconds = (finished - loaded) / 1e9,
        )
    catch
        HiGHS.Highs_destroy(highs)
        rethrow()
    end
end

function main(sizes = [1000]; repeats = 5, modes = MODES, batch_size = 65_536)
    for mode in modes
        highs, _ = construct(5, mode; batch_size)
        HiGHS.Highs_destroy(highs)
        for n in sizes, sample = 1:repeats
            highs, timings = construct(n, mode; batch_size)
            try
                println(
                    JSON.json((;
                        event = "sample",
                        mode,
                        size = n,
                        sample,
                        batch_size,
                        variables = HiGHS.Highs_getNumCol(highs),
                        constraints = HiGHS.Highs_getNumRow(highs),
                        timings...,
                    )),
                )
                flush(stdout)
            finally
                HiGHS.Highs_destroy(highs)
            end
        end
    end
    return
end

end # module MatrixHiGHSBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    sizes = isempty(ARGS) ? [1000] : parse.(Int, ARGS)
    MatrixHiGHSBenchmark.main(sizes)
end
