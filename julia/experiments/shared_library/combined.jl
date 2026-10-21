# One untrimmed JuliaC image containing the real JuMPy HiGHS backend and
# constructor exports. PythonCall is loaded later only when requested.
include(joinpath(@__DIR__, "..", "..", "juliac_entry.jl"))
include("constructors.jl")

module JuMPyCombinedProbe

import Main.JuMPyHiGHS
import Main.JuMPyMOIProbe
import MathOptInterface as MOI

@assert JuMPyHiGHS.JuMPyMOI === JuMPyMOIProbe.JuMPyMOI

# Keep a diagnostic reference to the set the compiled entry point consumed.
# No serialization or reconstruction of the set occurs at this boundary.
const LAST_SET = Ref{Any}(nothing)

# Validate experimental model/function handles against their owning roots
# before accessing them. Do not dereference arbitrary caller addresses.
function model_object(pointer::Ptr{Cvoid})
    Base.@lock JuMPyHiGHS.LOCK begin
        for model in keys(JuMPyHiGHS.KEEP_ALIVE)
            pointer_from_objref(model) == pointer && return model
        end
    end
    error("Unknown or released model")
end

function function_object(model::JuMPyHiGHS.ModelHandle, pointer::Ptr{Cvoid})
    for node in model.nodes
        pointer_from_objref(node) == pointer && return node[]
    end
    error("Function does not belong to this model")
end

Base.@ccallable function jumpy_probe_add_constraint(
    model_pointer::Ptr{Cvoid},
    function_pointer::Ptr{Cvoid},
    set_handle::UInt64,
)::Clonglong
    try
        model = model_object(model_pointer)
        func = JuMPyHiGHS.JuMPyMOI.simplify(function_object(model, function_pointer))
        set = JuMPyMOIProbe.get_object(set_handle)
        # This experiment deliberately tests only the existing LessThan set.
        set isa MOI.LessThan{Float64} || error("Expected MOI.LessThan{Float64}")
        ci = MOI.Utilities.normalize_and_add_constraint(model.optimizer, func, set)
        LAST_SET[] = set
        return ci.value
    catch
        # No Julia exception may unwind through the C ABI.
        return -1
    end
end

end # module JuMPyCombinedProbe
