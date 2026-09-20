module JuMPyHiGHS

# C entry points for the JuMPy juliac backend.
#
# JuMP models and expressions stay in Julia. Opaque expression handles
# let Python call JuMP's arithmetic and constraint-building methods, with
# GenOpt's bridge handling generated constraints. Nothing here is specific
# to HiGHS except the `Optimizer` constant below.
#
# Conventions across the C ABI:
#   - a model is an opaque pointer returned by jumpy_new_model; it stays
#     valid until jumpy_free_model, after which it must not be used
#   - expressions are opaque pointers built with jumpy_constant /
#     jumpy_variable / jumpy_apply; they belong to the model
#     that built them and are freed with it
#   - variables are 0-based column indices in the order they were added
#   - sets are checked UInt64 handles, independent of models, valid until
#     jumpy_free_set; zero is reserved for constructor errors
#   - objective sense: 0 = min, 1 = max
#   - entry points return -1 (NULL, NaN) on error, after printing to stderr
#     (a Julia exception must never propagate across the C boundary)

import GenOpt
import HiGHS
import JuMP
import MathOptInterface as MOI

include("JuMPyModel.jl")

# The only solver-specific line in this package.
const Optimizer = HiGHS.Optimizer

mutable struct ModelHandle
    model::JuMP.Model
    variables::Vector{JuMP.VariableRef}
    constraints::Vector{JuMP.ConstraintRef}
    # Roots the expression nodes handed out as pointers: the Julia GC
    # cannot see references held by the C caller.
    nodes::Vector{Base.RefValue{Any}}
end

# Same rooting, for the models themselves: alive until jumpy_free_model.
const KEEP_ALIVE = IdDict{ModelHandle,Nothing}()
const LOCK = ReentrantLock()

function _get(model::Ptr{Cvoid})
    model == C_NULL && error("Model pointer is NULL")
    return unsafe_pointer_to_objref(model)::ModelHandle
end

# Expression nodes (constants, JuMP expressions, GenOpt templates) are boxed in
# a Ref so that immutable values also get a stable pointer.
function _box(handle::ModelHandle, value)::Ptr{Cvoid}
    node = Base.RefValue{Any}(value)
    push!(handle.nodes, node)
    return pointer_from_objref(node)
end

function _unbox(node::Ptr{Cvoid})
    node == C_NULL && error("Expression pointer is NULL")
    return (unsafe_pointer_to_objref(node)::Base.RefValue{Any})[]
end

# Report without letting Julia exceptions cross the C ABI.
function _print_error(@nospecialize(err))
    print(stderr, "JuMPyHiGHS error: ")
    showerror(stderr, err)
    println(stderr)
    return
end

macro _catch(default, expr)
    quote
        try
            $(esc(expr))
        catch err
            _print_error(err)
            $(esc(default))
        end
    end
end

# -- Native scalar sets -------------------------------------------------------

# No Julia object pointers leave this registry. A set can be reused across
# models in this image, but never in another backend's Julia runtime.
const SETS = Dict{UInt64,MOI.AbstractScalarSet}()
const SET_LOCK = ReentrantLock()
const NEXT_SET = Ref{UInt64}(1)

function _get_set(id::UInt64)
    Base.@lock SET_LOCK begin
        haskey(SETS, id) || throw(ArgumentError("Invalid or released native MOI set"))
        return SETS[id]
    end
end

function _store_set(set::MOI.AbstractScalarSet)
    Base.@lock SET_LOCK begin
        id = NEXT_SET[]
        id != 0 || error("Native MOI set handle space exhausted")
        SETS[id] = set
        NEXT_SET[] += UInt64(1)  # never reuse released handles
        return id
    end
end

Base.@ccallable function jumpy_less_than(rhs::Cdouble)::UInt64
    @_catch UInt64(0) _store_set(MOI.LessThan(rhs))
end

Base.@ccallable function jumpy_greater_than(rhs::Cdouble)::UInt64
    @_catch UInt64(0) _store_set(MOI.GreaterThan(rhs))
end

Base.@ccallable function jumpy_equal_to(rhs::Cdouble)::UInt64
    @_catch UInt64(0) _store_set(MOI.EqualTo(rhs))
end

Base.@ccallable function jumpy_zero_one()::UInt64
    @_catch UInt64(0) _store_set(MOI.ZeroOne())
end

Base.@ccallable function jumpy_integer()::UInt64
    @_catch UInt64(0) _store_set(MOI.Integer())
end

Base.@ccallable function jumpy_free_set(id::UInt64)::Cint
    @_catch Cint(-1) begin
        Base.@lock SET_LOCK begin
            haskey(SETS, id) || throw(ArgumentError("Invalid or released native MOI set"))
            delete!(SETS, id)
        end
        Cint(0)
    end
end

# -- Model lifecycle ----------------------------------------------------------

Base.@ccallable function jumpy_new_model()::Ptr{Cvoid}
    @_catch C_NULL begin
        model = JuMPyModel.model(Optimizer())
        handle = ModelHandle(
            model, JuMP.VariableRef[], JuMP.ConstraintRef[], Base.RefValue{Any}[],
        )
        Base.@lock LOCK KEEP_ALIVE[handle] = nothing
        pointer_from_objref(handle)
    end
end

Base.@ccallable function jumpy_free_model(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        Base.@lock LOCK delete!(KEEP_ALIVE, _get(model))
        Cint(0)
    end
end

# -- Variables ----------------------------------------------------------------

# Returns the 0-based index of the first added JuMP variable.
# Bounds are constraints: pass a variable node to jumpy_add_constraint.
Base.@ccallable function jumpy_add_variables(
    model::Ptr{Cvoid},
    count::Clonglong,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        start = length(handle.variables)
        append!(handle.variables, JuMPyModel.add_variables(handle.model, count))
        Clonglong(start)
    end
end

# -- Native expression constructors -------------------------------------------

Base.@ccallable function jumpy_constant(
    model::Ptr{Cvoid},
    value::Cdouble,
)::Ptr{Cvoid}
    @_catch C_NULL _box(_get(model), value)
end

Base.@ccallable function jumpy_integer_constant(
    model::Ptr{Cvoid},
    value::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL _box(_get(model), value)
end

# JuMP.VariableRef of the 0-based column `index`.
Base.@ccallable function jumpy_variable(
    model::Ptr{Cvoid},
    index::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, handle.variables[index+1])
    end
end

# Call the ordinary Julia operator on the actual expression objects.
Base.@ccallable function jumpy_apply(
    model::Ptr{Cvoid},
    head::Cstring,
    args::Ptr{Ptr{Cvoid}},
    nargs::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        func = JuMPyModel.apply(
            Symbol(unsafe_string(head)),
            Any[_unbox(unsafe_load(args, k)) for k in 1:nargs],
        )
        _box(handle, func)
    end
end

# GenOpt's native iterator participates in its JuMP operator overloads.
Base.@ccallable function jumpy_iterator(
    model::Ptr{Cvoid},
    values::Ptr{Cdouble},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, JuMPyModel.iterator([unsafe_load(values, k) for k in 1:len]))
    end
end

Base.@ccallable function jumpy_integer_iterator(
    model::Ptr{Cvoid},
    values::Ptr{Clonglong},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, JuMPyModel.iterator([unsafe_load(values, k) for k in 1:len]))
    end
end

# JuMP references for the block beginning at 0-based column `offset`.
Base.@ccallable function jumpy_contiguous_variables(
    model::Ptr{Cvoid},
    offset::Clonglong,
    count::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, JuMPyModel.contiguous_variables(handle.model, offset, count))
    end
end

# A data vector, indexable (1-based) inside a template.
Base.@ccallable function jumpy_float_array(
    model::Ptr{Cvoid},
    values::Ptr{Cdouble},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, [unsafe_load(values, k) for k in 1:len])
    end
end

# -- Constraints --------------------------------------------------------------

# Scalar and generated constraints use the same JuMP construction path.
# Returns a positive model-local handle (MOI's bridged indices may be negative).
Base.@ccallable function jumpy_add_constraint(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
    set_id::UInt64,
)::Clonglong
    @_catch Clonglong(-1) begin
        set = _get_set(set_id)
        handle = _get(model)
        constraint = JuMPyModel.add_constraint(handle.model, _unbox(func), set)
        push!(handle.constraints, constraint)
        Clonglong(length(handle.constraints))
    end
end

# -- Objective ----------------------------------------------------------------

# MOI.set(MOI.ObjectiveSense()); 0 = min, 1 = max.
Base.@ccallable function jumpy_set_objective_sense(
    model::Ptr{Cvoid},
    sense::Cint,
)::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        moi_sense = sense == 0 ? MOI.MIN_SENSE : MOI.MAX_SENSE
        JuMPyModel.set_objective_sense(handle.model, moi_sense)
        Cint(0)
    end
end

# MOI.set(MOI.ObjectiveFunction{F}(), func).
Base.@ccallable function jumpy_set_objective_function(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
)::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        JuMPyModel.set_objective_function(handle.model, _unbox(func))
        Cint(0)
    end
end

# -- Solve and solution retrieval ----------------------------------------------

# Returns Int(MOI.TerminationStatusCode); MOI.OPTIMAL is 1.
Base.@ccallable function jumpy_optimize(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        JuMPyModel.optimize(handle.model)
        Cint(Integer(JuMP.termination_status(handle.model)))
    end
end

# Returns Int(MOI.ResultStatusCode) of the primal; MOI.FEASIBLE_POINT is 1.
Base.@ccallable function jumpy_primal_status(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        Cint(Integer(JuMPyModel.primal_status(handle.model)))
    end
end

# Writes the primal values of the first min(len, num variables) variables
# into `out`. Returns the number of values written.
Base.@ccallable function jumpy_get_values(
    model::Ptr{Cvoid},
    out::Ptr{Cdouble},
    len::Clonglong,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        n = min(len, length(handle.variables))
        values = JuMPyModel.get_values(handle.model, handle.variables[1:n])
        for k in 1:n
            unsafe_store!(out, values[k], k)
        end
        Clonglong(n)
    end
end

Base.@ccallable function jumpy_objective_value(model::Ptr{Cvoid})::Cdouble
    @_catch Cdouble(NaN) begin
        handle = _get(model)
        Cdouble(JuMPyModel.objective_value(handle.model))
    end
end

end # module JuMPyHiGHS
