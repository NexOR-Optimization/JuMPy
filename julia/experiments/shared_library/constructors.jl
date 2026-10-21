# Experimental constructor-only JuliaC image, not a production JuMPy backend.
module JuMPyMOIProbe

if isdefined(Main, :JuMPyHiGHS)
    # In the combined image both consumers use the very same module.
    import Main.JuMPyHiGHS: JuMPyMOI
else
    include(joinpath(@__DIR__, "..", "..", "src", "JuMPyMOI.jl"))
end
import .JuMPyMOI: MOI

# Handles are scoped to this image. They are NOT addresses of Julia objects.
# Keeping objects in this registry roots them across C calls and collections.
const OBJECTS = Dict{UInt64,Any}()
const NEXT_HANDLE = Ref{UInt64}(0)

function keep(object)::UInt64
    NEXT_HANDLE[] == typemax(UInt64) && error("Handle space exhausted")
    handle = (NEXT_HANDLE[] += 1)
    OBJECTS[handle] = object
    return handle
end

get_object(handle::UInt64) = OBJECTS[handle]

Base.@ccallable function jumpy_moi_new_less_than(rhs::Cdouble)::UInt64
    try
        return keep(JuMPyMOI.scalar_set(Val(0), rhs))
    catch
        return 0
    end
end

Base.@ccallable function jumpy_moi_new_variable(index::Clonglong)::UInt64
    try
        index > 0 || return 0
        return keep(MOI.VariableIndex(index))
    catch
        return 0
    end
end

Base.@ccallable function jumpy_moi_new_constant(value::Cdouble)::UInt64
    try
        return keep(value)
    catch
        return 0
    end
end

Base.@ccallable function jumpy_moi_new_add(lhs::UInt64, rhs::UInt64)::UInt64
    try
        return keep(JuMPyMOI.scalar_nonlinear(:+, Any[get_object(lhs), get_object(rhs)]))
    catch
        return 0
    end
end

Base.@ccallable function jumpy_moi_kind(handle::UInt64)::Cint
    object = get(OBJECTS, handle, nothing)
    object isa MOI.LessThan{Float64} && return 1
    object isa MOI.VariableIndex && return 2
    object isa Float64 && return 3
    object isa MOI.ScalarNonlinearFunction && return 4
    return -1
end

Base.@ccallable function jumpy_moi_bound(handle::UInt64)::Cdouble
    object = get(OBJECTS, handle, nothing)
    return object isa MOI.LessThan{Float64} ? object.upper : NaN
end

Base.@ccallable function jumpy_moi_arity(handle::UInt64)::Clonglong
    object = get(OBJECTS, handle, nothing)
    return object isa MOI.ScalarNonlinearFunction ? length(object.args) : -1
end

Base.@ccallable function jumpy_moi_release(handle::UInt64)::Cint
    haskey(OBJECTS, handle) || return -1
    delete!(OBJECTS, handle)
    return 0
end

Base.@ccallable function jumpy_moi_collect()::Cvoid
    GC.gc(true)
    return
end

# Diagnostic identity tokens only: the caller compares these addresses but
# must never dereference one using a different Julia runtime.
Base.@ccallable function jumpy_moi_type_token(kind::Cint)::Ptr{Cvoid}
    kind == 1 && return pointer_from_objref(MOI.LessThan{Float64})
    kind == 2 && return pointer_from_objref(MOI.VariableIndex)
    kind == 3 && return pointer_from_objref(Float64)
    kind == 4 && return pointer_from_objref(MOI.ScalarNonlinearFunction)
    return C_NULL
end

end # module JuMPyMOIProbe
