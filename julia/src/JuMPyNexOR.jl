module JuMPyNexOR

import MathOptInterface as MOI
import NexOR

# Restrict the cache to the scalar functions and rows exposed by the ABI.
MOI.Utilities.@model(
    ScalarModel,
    (),
    (MOI.LessThan, MOI.GreaterThan, MOI.EqualTo),
    (),
    (),
    (MOI.ScalarNonlinearFunction,),
    (MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction),
    (),
    (),
)

const Optimizer = MOI.Utilities.CachingOptimizer{NexOR.Optimizer,ScalarModel{Float64}}

MOI.supports(::ScalarModel, ::MOI.Silent) = true
function MOI.set(model::ScalarModel, ::MOI.Silent, value::Bool)
    model.ext[:jumpy_silent] = value
    return
end
MOI.get(model::ScalarModel, ::MOI.Silent) = get(model.ext, :jumpy_silent, false)::Bool
function MOI.get(model::ScalarModel, ::MOI.ListOfOptimizerAttributesSet)
    return haskey(model.ext, :jumpy_silent) ? MOI.AbstractOptimizerAttribute[MOI.Silent()] :
           MOI.AbstractOptimizerAttribute[]
end

function _copy_rows!(dest, src, map, ::Type{F}, ::Type{S}) where {F,S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        map[ci] = MOI.add_constraint(dest, MOI.Utilities.map_indices(map, f), set)
    end
    return
end

function _copy_rows!(dest, src, map, ::Type{F}) where {F}
    _copy_rows!(dest, src, map, F, MOI.LessThan{Float64})
    _copy_rows!(dest, src, map, F, MOI.GreaterThan{Float64})
    _copy_rows!(dest, src, map, F, MOI.EqualTo{Float64})
    return
end

function _copy_objective!(dest, src, map, ::Type{F}) where {F}
    f = MOI.get(src, MOI.ObjectiveFunction{F}())
    MOI.set(dest, MOI.ObjectiveFunction{F}(), MOI.Utilities.map_indices(map, f))
    return
end

# Copy only the attributes and scalar combinations available through the
# C ABI. The fixed calls avoid dispatching on runtime function/set types.
function MOI.copy_to(dest::NexOR.Optimizer, src::ScalarModel{Float64})
    MOI.empty!(dest)
    mock = dest.model
    map = MOI.Utilities.IndexMap()
    for vi in MOI.get(src, MOI.ListOfVariableIndices())
        map[vi] = MOI.add_variable(mock)
    end
    sense = MOI.get(src, MOI.ObjectiveSense())
    MOI.set(mock, MOI.ObjectiveSense(), sense)
    if sense != MOI.FEASIBILITY_SENSE
        F = MOI.get(src, MOI.ObjectiveFunctionType())
        if F == MOI.ScalarAffineFunction{Float64}
            _copy_objective!(mock, src, map, MOI.ScalarAffineFunction{Float64})
        elseif F == MOI.ScalarQuadraticFunction{Float64}
            _copy_objective!(mock, src, map, MOI.ScalarQuadraticFunction{Float64})
        else
            _copy_objective!(mock, src, map, MOI.ScalarNonlinearFunction)
        end
    end
    _copy_rows!(mock, src, map, MOI.VariableIndex)
    _copy_rows!(mock, src, map, MOI.VariableIndex, MOI.ZeroOne)
    _copy_rows!(mock, src, map, MOI.VariableIndex, MOI.Integer)
    _copy_rows!(mock, src, map, MOI.ScalarAffineFunction{Float64})
    _copy_rows!(mock, src, map, MOI.ScalarQuadraticFunction{Float64})
    _copy_rows!(mock, src, map, MOI.ScalarNonlinearFunction)
    return map
end

function _new_optimizer()
    optimizer = NexOR.Optimizer()
    MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("solver"),
        get(ENV, "NEXOR_SOLVER", "HiGHS"),
    )
    # NexOR accepts a whole model via copy_to, so keep incremental edits in
    # the scalar cache and copy them to the remote optimizer at optimize!.
    return MOI.Utilities.CachingOptimizer(ScalarModel{Float64}(), optimizer)
end

include("entrypoints.jl")
include("nexor_trim.jl")


end # module JuMPyNexOR
