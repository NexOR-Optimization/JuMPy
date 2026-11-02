module JuMPyNexOR

import JuMPy
import NexOR

# The remote solver is the optimizer's "solver" attribute, set by the user.
# The server accepts any set, so the sets of the C ABI are listed explicitly.
JuMPy.@entrypoints(
    NexOR.Optimizer,
    (
        MOI.EqualTo{Float64},
        MOI.GreaterThan{Float64},
        MOI.Integer,
        MOI.LessThan{Float64},
        MOI.ZeroOne,
    ),
)

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

# Solver namespaces add one inferred dictionary layer inside Any-valued
# options. Normalize it to JSON's statically supported dictionary shape.
function NexOR.JSON.applyany(
    style::NexOR.JSON.JSONWriteStyle,
    f::F,
    key::K,
    @nospecialize(value)
) where {F,K}
    if value isa Dict{String,Dict{String,Any}}
        object = Dict{String,Any}()
        for (name, options) in value
            object[name] = options
        end
        return NexOR.JSON._writenested(f, key, object)
    end
    return invoke(
        NexOR.JSON.applyany,
        Tuple{NexOR.JSON.JSONStyle,Any,Any,Any},
        style,
        f,
        key,
        value,
    )
end

end # module JuMPyNexOR
