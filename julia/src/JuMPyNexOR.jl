module JuMPyNexOR

import MathOptInterface as MOI
import NexOR

const Optimizer = NexOR.Optimizer

function _new_optimizer()
    optimizer = Optimizer()
    MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("solver"),
        get(ENV, "NEXOR_SOLVER", "HiGHS"),
    )
    return optimizer
end

include("entrypoints.jl")

# MockOptimizer scrambles MOI indices. Keep the actual variable references
# when GenOpt builds a generator instead of assuming consecutive indices.
struct IndexedVariables <: AbstractVector{NativeVariable}
    data::Vector{NativeVariable}
end
Base.size(array::IndexedVariables) = size(array.data)
Base.getindex(array::IndexedVariables, i::Int) = array.data[i]
GenOpt.to_generator(array::Vector{NativeVariable}) = IndexedVariables(array)
JuMP._is_real(::IndexedVariables) = true
JuMP.moi_function(array::IndexedVariables) = JuMP.index.(array.data)
function JuMP.check_belongs_to_model(array::IndexedVariables, model::JuMP.AbstractModel)
    for variable in array.data
        JuMP.check_belongs_to_model(variable, model)
    end
    return
end
function JuMP.moi_function(model::JuMP.ConcreteModel, array::IndexedVariables)
    JuMP.check_belongs_to_model(array, model)
    return JuMP.moi_function(array)
end

include("nexor_trim.jl")

end # module JuMPyNexOR
