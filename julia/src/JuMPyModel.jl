module JuMPyModel

# Shared source, loaded separately in the JuliaCall and JuliaC runtimes.
# JuMP and GenOpt own expression promotion and coefficient types. JuliaCall
# uses the ordinary cached/bridged JuMP model; the compiled backend uses the
# separate direct adapter in JuMPyDirect.jl.
import GenOpt
import JuMP
import MathOptInterface as MOI

function model(optimizer = nothing; add_bridges::Bool = true, kwargs...)
    result = JuMP.Model(optimizer; add_bridges, kwargs...)
    if add_bridges
        JuMP.add_bridge(result, GenOpt.FunctionGeneratorBridge)
        JuMP.add_bridge(result, GenOpt.PolynomialFunctionBridge)
    end
    return result
end

add_variables(model, count) = JuMP.@variable(model, [1:count])

# The array of variables of dimensions `dims` constrained to belong to `set`
# on creation, as `@variable(model, [1:m, 1:n] in set)`.
function add_constrained_variables(model, set, dims::Int...)
    info = JuMP.VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false)
    scalars = [JuMP.ScalarVariable(info) for _ in CartesianIndices(dims)]
    # Anonymous variables, as `@variable(model, [1:m, 1:n] in set)`.
    names = fill("", dims)
    return JuMP.add_variable(model, JuMP.build_variable(error, scalars, set), names)
end

# `array[indices...]` for 1-based `(first, last)` pairs, where a scalar index
# has `last == typemin(Int)`, as the compiled jumpy_getindex.
function getindex_bounds(array, bounds::Int...)
    indices = ntuple(length(bounds) ÷ 2) do k
        first, last = bounds[2k-1], bounds[2k]
        return last == typemin(Int) ? first : first:last
    end
    return array[indices...]
end

# A data vector, or a column-major matrix of `rows` and `columns`.
function data_array(
    values::AbstractVector,
    rows::Int,
    columns::Int,
    ndims::Int,
    integer::Bool,
)
    data = integer ? Vector{Int}(values) : Vector{Float64}(values)
    return ndims == 1 ? data : reshape(data, rows, columns)
end

concatenate(args::Vector{Any}) = vcat(args...)
nonlinear(head::Symbol, args::Vector{Any}) = JuMP.NonlinearExpr(head, args)
lazy_sum(template::GenOpt.ExprTemplate) = GenOpt.LazySum(template)
# Keep references to the native functions: a trimmed image need not retain
# Base bindings looked up only by a runtime Symbol.
const _OPERATORS = Dict{Symbol,Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => ^,
    :getindex => getindex,
    :sin => sin,
    :cos => cos,
    :exp => exp,
    :log => log,
    :sqrt => sqrt,
    :abs => abs,
    :lazy_sum => lazy_sum,
)
apply(op::Symbol, args::Vector{Any}) = _OPERATORS[op](args...)
iterator(values) = GenOpt.iterator(values)

function add_constraint(model, func, set)
    # build_constraint normalizes constants in-place, as in @constraint.
    constraint = JuMP.build_constraint(error, JuMP._MA.copy_if_mutable(func), set)
    return JuMP.add_constraint(model, constraint)
end

set_objective_sense(model, sense) = JuMP.set_objective_sense(model, sense)
set_objective_function(model, func) = JuMP.set_objective_function(model, func)
optimize(model) = JuMP.optimize!(model)
value(func) = JuMP.value(func)
objective_value(model) = JuMP.objective_value(model)
has_values(model) = JuMP.has_values(model)
has_duals(model) = JuMP.has_duals(model)
termination_status(model) = JuMP.termination_status(model)

set_attribute(model, name::String, value) = JuMP.set_attribute(model, name, value)

function set_time_limit_sec(model, seconds::Float64)
    if isnan(seconds)
        JuMP.unset_time_limit_sec(model)
    else
        isfinite(seconds) && seconds >= 0 ||
            throw(ArgumentError("Time limit must be a finite nonnegative number"))
        JuMP.set_time_limit_sec(model, seconds)
    end
    return
end

function dual(model, constraint)
    JuMP.owner_model(constraint) === model ||
        throw(ArgumentError("Constraint does not belong to this model"))
    has_duals(model) || error("No dual solution is available")
    value = JuMP.dual(constraint)
    value isa Real || error("Expected a scalar constraint dual")
    return value
end

end # module JuMPyModel
