module JuMPyModel

# Shared source, loaded separately in the JuliaCall and JuliaC runtimes.
# JuMP and GenOpt own expression promotion and coefficient types. JuliaCall
# uses the ordinary cached/bridged JuMP model; the compiled backend uses the
# separate direct adapter in JuMPyDirect.jl.
import GenOpt
import JuMP
import MathOptInterface as MOI

function model(optimizer)
    result = JuMP.Model(() -> optimizer)
    JuMP.add_bridge(result, GenOpt.FunctionGeneratorBridge)
    JuMP.set_silent(result)
    return result
end

add_variables(model, count) = JuMP.@variable(model, [1:count])
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

end # module JuMPyModel
