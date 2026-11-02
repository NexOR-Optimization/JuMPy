# Bottom-up LQCP experiment: GenOpt plans feeding native HiGHS row batches.
module NativeGenOptBenchmark

import GenOpt
import HiGHS
import JSON
import JuMP
import MathOptInterface as MOI
include("direct_highs.jl")
include("grouped_models.jl")
const Direct = DirectHiGHSBenchmark
const H = HiGHS.HighsInt
const F = MOI.ScalarAffineFunction{Float64}
const S = MOI.EqualTo{Float64}
const CI = MOI.ConstraintIndex{F,S}

# This is an experimental adapter for this formulation, not a public backend.
mutable struct NativeRows{B} <: MOI.AbstractOptimizer
    backend::B
    columns::Bool
    bulk_variables::Bool
    bulk_bounds::Bool
    nv::Int
    nr::Int
    calls::Int
    pack_ns::UInt64
    native_ns::UInt64
    bookkeeping_ns::UInt64
    variables_ns::UInt64
    bounds_ns::UInt64
    objective_ns::UInt64
    expansion_ns::UInt64
end
NativeRows(backend, nv; columns = false, bulk_variables = false, bulk_bounds = false) =
    NativeRows{typeof(backend)}(
        backend,
        columns,
        bulk_variables,
        bulk_bounds,
        nv,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

column(model::NativeRows{Ptr{Cvoid}}, v) = H(v.value - 1)
column(model::NativeRows{HiGHS.Optimizer}, v) = HiGHS.column(model.backend, v)

function record_rows(model, rhs)
    started = time_ns()
    indices = Vector{CI}(undef, length(rhs))
    for i in eachindex(rhs)
        if model.backend isa HiGHS.Optimizer
            key = HiGHS.CleverDicts.add_item(
                model.backend.affine_constraint_info,
                HiGHS._ConstraintInfo(S(rhs[i])),
            )
            model.backend.affine_constraint_info[key].row = H(model.nr + i - 1)
            indices[i] = CI(key.value)
        else
            indices[i] = CI(model.nr + i)
        end
    end
    model.nr += length(rhs)
    model.bookkeeping_ns += time_ns() - started
    return indices
end

function add_rows(model, starts, indices, values, rhs)
    started = time_ns()
    Direct.check(
        HiGHS.Highs_addRows(
            model.backend,
            length(rhs),
            rhs,
            rhs,
            length(values),
            starts,
            indices,
            values,
        ),
    )
    model.native_ns += time_ns() - started
    model.calls += 1
    return record_rows(model, rhs)
end

function MOI.add_constraints(model::NativeRows, functions::Vector{F}, sets::Vector{S})
    length(functions) == length(sets) || throw(DimensionMismatch("Functions and sets"))
    started = time_ns()
    canonical = map(functions) do f
        MOI.throw_if_scalar_and_constant_not_zero(f, S)
        for term in f.terms
            1 <= term.variable.value <= model.nv || throw(MOI.InvalidIndex(term.variable))
        end
        return MOI.Utilities.is_canonical(f) ? f : MOI.Utilities.canonical(f)
    end
    starts = Vector{H}(undef, length(functions) + 1)
    nnz = sum(f -> length(f.terms), canonical; init = 0)
    indices, values = Vector{H}(undef, nnz), Vector{Float64}(undef, nnz)
    rhs = [set.value for set in sets]
    position = 1
    for row in eachindex(canonical)
        starts[row] = position - 1
        for term in canonical[row].terms
            indices[position] = column(model, term.variable)
            values[position] = term.coefficient
            position += 1
        end
    end
    starts[end] = position - 1
    model.pack_ns += time_ns() - started
    return add_rows(model, starts, indices, values, rhs)
end

# Typed term-column loops, just like GenOpt's affine-function fill kernels.
function fill_column!(
    indices,
    values,
    model,
    slot,
    width,
    count,
    offset,
    coefficients,
    variables,
)
    for row = 1:count
        position = (row - 1) * width + slot
        variable = GenOpt._batch_at(variables, row + offset)
        1 <= variable.value <= model.nv || throw(MOI.InvalidIndex(variable))
        indices[position] = column(model, variable)
        values[position] = GenOpt._batch_at(coefficients, row + offset)
    end
    return
end

# Sort and merge each short stencil in place. No per-row function or term vector.
function compact_rows!(starts, indices, values, count, width)
    position = 1
    for row = 1:count
        first = position
        starts[row] = position - 1
        for entry = ((row-1)*width+1):(row*width)
            index, value = indices[entry], values[entry]
            target = first
            while target < position && indices[target] < index
                target += 1
            end
            if target < position && indices[target] == index
                values[target] += value
            else
                for k = position:-1:(target+1)
                    indices[k], values[k] = indices[k-1], values[k-1]
                end
                indices[target], values[target] = index, value
                position += 1
            end
        end
        stop = position
        position = first
        for k = first:(stop-1)
            if !iszero(values[k])
                indices[position], values[position] = indices[k], values[k]
                position += 1
            end
        end
    end
    starts[end] = position - 1
    resize!(indices, position - 1)
    resize!(values, position - 1)
    return
end

function add_plan(model::NativeRows, plan, set::S, rows; batch_size = 65_536)
    batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
    isempty(plan.quadratic) || error("Only affine row plans are supported")
    constraints = Vector{CI}(undef, rows)
    for first = 1:batch_size:rows
        count, offset = min(batch_size, rows - first + 1), first - 1
        if model.columns
            started = time_ns()
            width = length(plan.affine)
            starts = Vector{H}(undef, count + 1)
            indices, values =
                Vector{H}(undef, count * width), Vector{Float64}(undef, count * width)
            constants = GenOpt._batch_kernel_column(Float64, plan.constant)
            rhs = [set.value - GenOpt._batch_at(constants, row + offset) for row = 1:count]
            for (slot, term) in enumerate(plan.affine)
                fill_column!(
                    indices,
                    values,
                    model,
                    slot,
                    width,
                    count,
                    offset,
                    GenOpt._batch_kernel_column(Float64, term.coefficient),
                    GenOpt._batch_kernel_column(MOI.VariableIndex, term.variable),
                )
            end
            compact_rows!(starts, indices, values, count, width)
            model.pack_ns += time_ns() - started
            added = add_rows(model, starts, indices, values, rhs)
        else
            started = time_ns()
            functions = GenOpt._batch_functions(F, count, length(plan.affine), 0)
            GenOpt._batch_fill_constants!(
                functions,
                GenOpt._batch_kernel_column(Float64, plan.constant),
                offset,
            )
            for (slot, term) in enumerate(plan.affine)
                GenOpt._batch_fill_affine!(
                    functions,
                    slot,
                    GenOpt._batch_kernel_column(Float64, term.coefficient),
                    GenOpt._batch_kernel_column(MOI.VariableIndex, term.variable),
                    offset,
                )
            end
            sets = Vector{S}(undef, count)
            for row in eachindex(functions)
                functions[row], sets[row] = MOI.Utilities.normalize_constant(
                    functions[row],
                    set;
                    allow_modify_function = true,
                )
            end
            model.expansion_ns += time_ns() - started
            added = MOI.add_constraints(model, functions, sets)
        end
        copyto!(constraints, first, added, 1, count)
    end
    return constraints
end

# Use the real public GenOpt planner, specializing only its insertion stage for
# this experimental adapter. Works with both scalar and bulk GenOpt versions.
function GenOpt._batch_add_constraints(
    model::NativeRows,
    plan::GenOpt._BatchPolynomial{Float64},
    ::Type{F},
    set::S,
    rows,
)
    return add_plan(model, plan, set, rows)
end

snf(head, args...) = MOI.ScalarNonlinearFunction(head, Any[args...])
function generators(n)
    n >= 3 || throw(ArgumentError("LQCP size must be at least three"))
    side, dx, dt = n + 1, 1.0 / n, 1.58 / n
    y = GenOpt.ContiguousArrayOfVariables(0, (side^2,))
    u = GenOpt.ContiguousArrayOfVariables(side^2, (n,))
    t, j = GenOpt.IteratorIndex(1), GenOpt.IteratorIndex(2)
    current = snf(:+, t, snf(:*, side, j), 1)
    at(index) = snf(:getindex, y, index)
    next = snf(:+, current, 1)
    pde = snf(
        :-,
        snf(:*, dx^2, snf(:-, at(next), at(current))),
        snf(
            :*,
            0.5dt,
            snf(
                :+,
                at(snf(:-, current, side)),
                snf(:*, -2, at(current)),
                at(snf(:+, current, side)),
                at(snf(:-, next, side)),
                snf(:*, -2, at(next)),
                at(snf(:+, next, side)),
            ),
        ),
    )
    initial = at(snf(:+, snf(:*, side, t), 1))
    left = snf(
        :+,
        at(snf(:+, t, 2side, 1)),
        snf(:*, -4, at(snf(:+, t, side, 1))),
        snf(:*, 3, at(snf(:+, t, 1))),
    )
    right_index = snf(:+, t, n * side, 1)
    right = snf(
        :-,
        snf(
            :+,
            at(snf(:+, t, (n-2)*side, 1)),
            snf(:*, -4, at(snf(:+, t, (n-1)*side, 1))),
            snf(:*, 3, at(right_index)),
        ),
        snf(:*, 2dx, snf(:-, snf(:getindex, u, t), at(right_index))),
    )
    make(expr, ranges...) = GenOpt.FunctionGenerator{F}(
        expr,
        GenOpt.Iterator[GenOpt.Iterator(collect(r)) for r in ranges],
    )
    return [
        make(pde, 0:(n-1), 1:(n-1)),
        make(initial, 0:n),
        make(left, 1:n),
        make(right, 1:n),
    ]
end

# Forward ordinary JuMP variable/bound/objective operations to HiGHS.jl, while
# recording their costs. Row insertion remains the adapter's native batch path.
MOI.is_empty(model::NativeRows{HiGHS.Optimizer}) = MOI.is_empty(model.backend)
MOI.supports_incremental_interface(::NativeRows) = true
MOI.supports(model::NativeRows, attr::MOI.AbstractOptimizerAttribute) =
    MOI.supports(model.backend, attr)
MOI.supports(model::NativeRows, attr::MOI.AbstractModelAttribute) =
    MOI.supports(model.backend, attr)
MOI.supports_constraint(
    model::NativeRows,
    ::Type{G},
    ::Type{T},
) where {G<:MOI.AbstractFunction,T<:MOI.AbstractSet} =
    MOI.supports_constraint(model.backend, G, T)
MOI.is_valid(model::NativeRows, index::MOI.VariableIndex) =
    MOI.is_valid(model.backend, index)
MOI.get(model::NativeRows, attr::MOI.AbstractModelAttribute) = MOI.get(model.backend, attr)
MOI.set(model::NativeRows, attr::MOI.AbstractOptimizerAttribute, value) =
    MOI.set(model.backend, attr, value)
function MOI.set(model::NativeRows, attr::MOI.AbstractModelAttribute, value)
    started = time_ns()
    MOI.set(model.backend, attr, value)
    model.objective_ns += time_ns() - started
    return
end
function MOI.add_variables(model::NativeRows, count::Int)
    started = time_ns()
    if model.bulk_variables
        count >= 0 || throw(ArgumentError("Variable count must be nonnegative"))
        first_column = HiGHS.Highs_getNumCol(model.backend)
        Direct.check(
            HiGHS.Highs_addCols(
                model.backend,
                count,
                zeros(count),
                fill(-Inf, count),
                fill(Inf, count),
                0,
                C_NULL,
                C_NULL,
                C_NULL,
            ),
        )
        variables = Vector{MOI.VariableIndex}(undef, count)
        for i = 1:count
            info = HiGHS._VariableInfo(MOI.VariableIndex(0), H(first_column + i - 1))
            index = HiGHS.CleverDicts.add_item(model.backend.variable_info, info)
            info.index = variables[i] = index
        end
    else
        variables = MOI.add_variables(model.backend, count)
    end
    model.nv += count
    model.variables_ns += time_ns() - started
    return variables
end
function MOI.add_variable(model::NativeRows)
    started = time_ns()
    variable = MOI.add_variable(model.backend)
    model.nv += 1
    model.variables_ns += time_ns() - started
    return variable
end
function MOI.add_constraints(
    model::NativeRows,
    variables::Vector{MOI.VariableIndex},
    sets::Vector{T},
) where {T<:MOI.AbstractScalarSet}
    started = time_ns()
    if model.bulk_bounds && T <: Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64}}
        length(variables) == length(sets) || throw(DimensionMismatch("Variables and sets"))
        columns, lower, upper =
            Vector{H}(undef, length(sets)), zeros(length(sets)), zeros(length(sets))
        indices = Vector{MOI.ConstraintIndex{MOI.VariableIndex,T}}(undef, length(sets))
        for i in eachindex(variables)
            info = HiGHS._info(model.backend, variables[i])
            HiGHS._update_info(info, sets[i])
            columns[i], lower[i], upper[i] = info.column, info.lower, info.upper
            indices[i] = MOI.ConstraintIndex{MOI.VariableIndex,T}(variables[i].value)
        end
        Direct.check(
            HiGHS.Highs_changeColsBoundsBySet(
                model.backend,
                length(sets),
                columns,
                lower,
                upper,
            ),
        )
    else
        indices = MOI.add_constraints(model.backend, variables, sets)
    end
    model.bounds_ns += time_ns() - started
    return indices
end

# Only this local benchmark's JuMP adapter changes variable-block creation.
# Return ordinary JuMP variable references, keeping HiGHS.jl's metadata intact.
function GroupedPaperModels.JuMPyDirect.add_variables(
    model::JuMP.ConcreteModel{Float64,<:NativeRows},
    count,
)
    adapter = JuMP.backend(model)
    if !adapter.bulk_variables
        return invoke(
            GroupedPaperModels.JuMPyDirect.add_variables,
            Tuple{Any,Any},
            model,
            count,
        )
    end
    indices = MOI.add_variables(adapter, count)
    model.is_model_dirty = true
    V = JuMP.variable_ref_type(model)
    return [V(model, index) for index in indices]
end

function setup(data, mapped; solve = false)
    backend = mapped ? HiGHS.Optimizer() : HiGHS.Highs_create()
    try
        Direct.check(HiGHS.Highs_setBoolOptionValue(backend, "output_flag", false))
        Direct.check(HiGHS.Highs_setIntOptionValue(backend, "threads", 1))
        Direct.check(HiGHS.Highs_setStringOptionValue(backend, "presolve", "off"))
        solve || Direct.check(HiGHS.Highs_setDoubleOptionValue(backend, "time_limit", 0.0))
        model = NativeRows(backend, 0)
        mapped && MOI.add_variables(model, data.nv)
        model.nv = data.nv
        Direct.check(
            HiGHS.Highs_passModel(
                backend,
                data.nv,
                0,
                0,
                length(data.qvalues),
                HiGHS.kHighsMatrixFormatRowwise,
                HiGHS.kHighsHessianFormatTriangular,
                HiGHS.kHighsObjSenseMinimize,
                data.offset,
                data.costs,
                data.lower,
                data.upper,
                Float64[],
                Float64[],
                H[0],
                H[],
                Float64[],
                data.qstarts,
                data.qindices,
                data.qvalues,
                C_NULL,
            ),
        )
        return model
    catch
        dispose(backend)
        rethrow()
    end
end

dispose(backend::Ptr{Cvoid}) = HiGHS.Highs_destroy(backend)
dispose(backend::HiGHS.Optimizer) = finalize(backend)

const MODES = (
    "native-arrays",
    "native-functions",
    "genopt-functions",
    "genopt-columns",
    "genopt-mapped",
    "grouped-functions",
    "grouped-columns",
    "grouped-bounds",
    "grouped-bulk",
    "grouped-stock",
)

function construct(n, mode; solve = false)
    mode in MODES || throw(ArgumentError("Unknown mode: $mode"))
    n >= 3 || throw(ArgumentError("LQCP size must be at least three"))
    started = time_ns()
    grouped = startswith(mode, "grouped-")
    build_started = UInt64(0)
    model = nothing
    try
        if grouped
            backend = HiGHS.Optimizer()
            model = NativeRows(
                backend,
                0;
                columns = mode in ("grouped-columns", "grouped-bounds", "grouped-bulk"),
                bulk_bounds = mode in ("grouped-bounds", "grouped-bulk"),
                bulk_variables = mode == "grouped-bulk",
            )
            MOI.set(backend, MOI.Silent(), true)
            MOI.set(backend, MOI.RawOptimizerAttribute("threads"), 1)
            MOI.set(backend, MOI.RawOptimizerAttribute("presolve"), "off")
            solve || MOI.set(backend, MOI.TimeLimitSec(), 0.0)
            jump = JuMP.concrete_direct_model(mode == "grouped-stock" ? backend : model)
            JuMP.set_string_names_on_creation(jump, false)
            build_started = time_ns()
            GroupedPaperModels.lqcp(jump, n)
            model.nr = HiGHS.Highs_getNumRow(backend)
        else
            data = Direct.lqcp_data(n)
            model = setup(data, mode == "genopt-mapped"; solve)
            model.columns = mode == "genopt-columns"
            build_started = time_ns()
            if mode in ("native-arrays", "native-functions")
                for first = 1:65_536:data.nr
                    rows = first:min(first+65_535, data.nr)
                    if mode == "native-arrays"
                        initial, stop =
                            Int(data.starts[first])+1, Int(data.starts[last(rows)+1])
                        t = time_ns()
                        starts = data.starts[first:(last(rows)+1)] .- data.starts[first]
                        model.pack_ns += time_ns() - t
                        add_rows(
                            model,
                            starts,
                            view(data.columns, initial:stop),
                            view(data.values, initial:stop),
                            view(data.rhs, rows),
                        )
                    else
                        functions = Vector{F}(undef, length(rows))
                        for (i, row) in enumerate(rows)
                            terms = [
                                MOI.ScalarAffineTerm(
                                    data.values[k],
                                    MOI.VariableIndex(data.columns[k]+1),
                                ) for k = (data.starts[row]+1):data.starts[row+1]
                            ]
                            functions[i] = F(terms, 0.0)
                        end
                        MOI.add_constraints(model, functions, S.(data.rhs[rows]))
                    end
                end
            else
                for generator in generators(n)
                    GenOpt.add_constraints(model, generator, S(0.0))
                end
            end
        end
        loaded = time_ns()
        if grouped
            MOI.optimize!(model.backend)
        else
            Direct.check(HiGHS.Highs_run(model.backend))
        end
        finished = time_ns()
        return model,
        (;
            seconds = (finished-started)/1e9,
            setup_seconds = (build_started-started)/1e9,
            build_seconds = (loaded-build_started)/1e9,
            pack_seconds = model.pack_ns/1e9,
            native_seconds = model.native_ns/1e9,
            bookkeeping_seconds = model.bookkeeping_ns/1e9,
            variables_seconds = model.variables_ns/1e9,
            bounds_seconds = model.bounds_ns/1e9,
            objective_seconds = model.objective_ns/1e9,
            run_seconds = (finished-loaded)/1e9,
            expansion_seconds = model.expansion_ns/1e9,
            calls = model.calls,
        )
    catch
        model === nothing || dispose(model.backend)
        rethrow()
    end
end

function main(sizes = [1000]; repeats = 5, modes = MODES)
    for mode in modes
        model, _ = construct(first(sizes), mode)
        dispose(model.backend)
        model = nothing
        for n in sizes, sample = 1:repeats
            GC.gc(true)
            model, timings = construct(n, mode)
            try
                println(
                    JSON.json((;
                        event = "sample",
                        mode,
                        size = n,
                        sample,
                        variables = HiGHS.Highs_getNumCol(model.backend),
                        constraints = model.nr,
                        timings...,
                    )),
                )
                flush(stdout)
            finally
                dispose(model.backend)
            end
            model = nothing
        end
    end
    return
end

end # module NativeGenOptBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    NativeGenOptBenchmark.main(isempty(ARGS) ? [1000] : parse.(Int, ARGS))
end
