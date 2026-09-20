module JuMPyModel

# Shared source, loaded separately in the JuliaCall and JuliaC runtimes.
# JuMP and GenOpt own expression promotion, coefficient types, and bridges.
import GenOpt
import JuMP
import MathOptInterface as MOI

function model(optimizer)
    backend = MOI.Bridges.full_bridge_optimizer(optimizer, Float64)
    MOI.Bridges.add_bridge(backend, GenOpt.FunctionGeneratorBridge{Float64})
    result = JuMP.direct_model(backend)
    JuMP.set_silent(result)
    return result
end

add_variables(model, count) = JuMP.@variable(model, [1:count])
apply(op::Symbol, args::Vector{Any}) = getfield(Base, op)(args...)
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

end # module JuMPyModel
