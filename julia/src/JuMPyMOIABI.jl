module JuMPyMOIABI

# Native MOI values owned by this Julia image, independently of any model.
# C callers receive checked integer handles, never pointers to Julia objects.
# A handle stays valid until released; handles are never reused. JuliaCall
# may borrow the actual object with get_object only in this same runtime.
import MathOptInterface as MOI
import ..JuMPyMOI
import ..JuMPyHiGHS: @_catch

const NativeObject = Union{
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
    MOI.EqualTo{Float64},
    MOI.ZeroOne,
    MOI.Integer,
    MOI.Nonpositives,
    MOI.Nonnegatives,
    MOI.Zeros,
    Float64,
    MOI.VariableIndex,
    MOI.ScalarNonlinearFunction,
}
const FunctionNode = Union{Float64,MOI.VariableIndex,MOI.ScalarNonlinearFunction}
const OBJECTS = Dict{UInt64,NativeObject}()
const OBJECT_LOCK = ReentrantLock()
const NEXT_HANDLE = Ref{UInt64}(1)

"""Root a supported native MOI object and return a new owned handle."""
function keep(value::NativeObject)::UInt64
    Base.@lock OBJECT_LOCK begin
        handle = NEXT_HANDLE[]
        handle != 0 || error("Native MOI handle space exhausted")
        OBJECTS[handle] = value
        NEXT_HANDLE[] += UInt64(1)
        return handle
    end
end

keep(@nospecialize(value)) =
    throw(ArgumentError("Object type is unsupported by the compiled MOI constructor ABI"))

"""Borrow an actual Julia object from this runtime, rejecting stale handles."""
function get_object(handle::UInt64)::NativeObject
    Base.@lock OBJECT_LOCK begin
        haskey(OBJECTS, handle) || throw(ArgumentError("Invalid native MOI handle"))
        return OBJECTS[handle]
    end
end

# Kinds are stable ABI values, not Julia type IDs:
# 1 LessThan, 2 GreaterThan, 3 EqualTo, 4 ZeroOne, 5 Integer,
# 6 Nonpositives, 7 Nonnegatives, 8 Zeros, 9 Float64,
# 10 VariableIndex, 11 ScalarNonlinearFunction. -1 means an invalid handle.
Base.@ccallable function jumpy_moi_kind(handle::UInt64)::Cint
    @_catch Cint(-1) begin
        value = get_object(handle)
        if value isa MOI.LessThan{Float64}
            Cint(1)
        elseif value isa MOI.GreaterThan{Float64}
            Cint(2)
        elseif value isa MOI.EqualTo{Float64}
            Cint(3)
        elseif value isa MOI.ZeroOne
            Cint(4)
        elseif value isa MOI.Integer
            Cint(5)
        elseif value isa MOI.Nonpositives
            Cint(6)
        elseif value isa MOI.Nonnegatives
            Cint(7)
        elseif value isa MOI.Zeros
            Cint(8)
        elseif value isa Float64
            Cint(9)
        elseif value isa MOI.VariableIndex
            Cint(10)
        else
            Cint(11)
        end
    end
end

# Scalar sense tags match jumpy_add_constraint: 0 <=, 1 >=, 2 ==,
# 3 binary, 4 integer. The right-hand side is ignored for the last two.
Base.@ccallable function jumpy_moi_scalar_set(sense::Cint, rhs::Cdouble)::UInt64
    @_catch UInt64(0) keep(JuMPyMOI.scalar_set(sense, rhs))
end

# Vector sense tags: 0 nonpositive, 1 nonnegative, 2 zero.
Base.@ccallable function jumpy_moi_vector_set(
    sense::Cint,
    dimension::Clonglong,
)::UInt64
    @_catch UInt64(0) begin
        dimension >= 0 || throw(ArgumentError("Set dimension must be nonnegative"))
        keep(JuMPyMOI.vector_set(sense, dimension))
    end
end

Base.@ccallable function jumpy_moi_constant(value::Cdouble)::UInt64
    @_catch UInt64(0) keep(value)
end

# This is an MOI index (1-based), unlike the existing model-column ABI.
Base.@ccallable function jumpy_moi_variable(index::Clonglong)::UInt64
    @_catch UInt64(0) begin
        index > 0 || throw(ArgumentError("MOI variable indices must be positive"))
        keep(MOI.VariableIndex(index))
    end
end

# The argument objects are retained by the constructed expression itself;
# callers can immediately release argument handles after construction.
Base.@ccallable function jumpy_moi_scalar_nonlinear(
    head::Cstring,
    args::Ptr{UInt64},
    nargs::Clonglong,
)::UInt64
    @_catch UInt64(0) begin
        head != C_NULL || throw(ArgumentError("Nonlinear head is NULL"))
        nargs >= 0 || throw(ArgumentError("Argument count must be nonnegative"))
        nargs == 0 || args != C_NULL || throw(ArgumentError("Argument array is NULL"))
        values = Vector{Any}(undef, nargs)
        for k in 1:nargs
            value = get_object(unsafe_load(args, k))
            value isa FunctionNode || throw(ArgumentError("Nonlinear arguments must be function nodes"))
            values[k] = value
        end
        keep(JuMPyMOI.scalar_nonlinear(Symbol(unsafe_string(head)), values))
    end
end

Base.@ccallable function jumpy_moi_release(handle::UInt64)::Cint
    @_catch Cint(-1) begin
        Base.@lock OBJECT_LOCK begin
            haskey(OBJECTS, handle) || throw(ArgumentError("Invalid native MOI handle"))
            delete!(OBJECTS, handle)
        end
        Cint(0)
    end
end

end # module JuMPyMOIABI
