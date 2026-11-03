# C entry points for the JuMPy juliac backend.
#
# JuMP models and expressions stay in Julia. Opaque expression handles
# let Python call JuMP's arithmetic and constraint-building methods, with
# GenOpt expanding generated constraints directly. `JuMPy.@entrypoints`
# defines `Optimizer` and `SET_TYPES` before including this file in the module
# of the backend.
#
# Conventions across the C ABI:
#   - a model is an opaque pointer returned by jumpy_new_model; it stays
#     valid until jumpy_free_model, after which it must not be used
#   - expressions are opaque pointers built with jumpy_constant /
#     jumpy_variables / jumpy_apply; they belong to the model
#     that built them and are freed with it
#   - variable arrays and individual variables use the same expression handles
#   - sets are checked UInt64 handles, independent of models, valid until
#     jumpy_free_set; zero is reserved for constructor errors. The sets that
#     jumpy_new_set constructs are described by jumpy_registry
#   - objective sense: 0 = min, 1 = max
#   - entry points return -1 (NULL, NaN) on error, after printing to stderr
#     (a Julia exception must never propagate across the C boundary)

const JuMPyModel = JuMPy.JuMPyModel
const JuMPyDirect = JuMPy.JuMPyDirect

const NativeModel = JuMP.ConcreteModel{Float64,Optimizer,Nothing}
const NativeVariable = JuMP.variable_ref_type(NativeModel)
const NativeAffine = JuMP.GenericAffExpr{Float64,NativeVariable}
const NativeQuadratic = JuMP.GenericQuadExpr{Float64,NativeVariable}
const NativeNonlinear = JuMP.GenericNonlinearExpr{NativeVariable}
const NativePolynomial = Union{
    GenOpt.PolynomialExpr{Int64,NativeVariable},
    GenOpt.PolynomialExpr{Float64,NativeVariable},
    GenOpt.PolynomialExpr{NativeVariable,NativeVariable},
    GenOpt.PolynomialExpr{NativeAffine,NativeVariable},
    GenOpt.PolynomialExpr{NativeQuadratic,NativeVariable},
    GenOpt.PolynomialExpr{Int64,JuMP.VariableRef},
    GenOpt.PolynomialExpr{Float64,JuMP.VariableRef},
}

const NativeLazySum = Union{
    GenOpt.LazySum{Int64,NativeVariable},
    GenOpt.LazySum{Float64,NativeVariable},
    GenOpt.LazySum{NativeVariable,NativeVariable},
    GenOpt.LazySum{NativeAffine,NativeVariable},
    GenOpt.LazySum{NativeQuadratic,NativeVariable},
    GenOpt.LazySum{NativeNonlinear,NativeVariable},
    GenOpt.LazySum{Int64,JuMP.VariableRef},
    GenOpt.LazySum{Float64,JuMP.VariableRef},
}
const NativeScalar = Union{
    Int64,
    Float64,
    NativeVariable,
    NativeAffine,
    NativeQuadratic,
    NativeNonlinear,
    JuMP.NonlinearExpr,
    NativeLazySum,
    NativePolynomial,
}
const NativeTemplate = Union{
    GenOpt.ExprTemplate{Int64,NativeVariable},
    GenOpt.ExprTemplate{Float64,NativeVariable},
    GenOpt.ExprTemplate{NativeVariable,NativeVariable},
    GenOpt.ExprTemplate{NativeAffine,NativeVariable},
    GenOpt.ExprTemplate{NativeQuadratic,NativeVariable},
    GenOpt.ExprTemplate{NativeNonlinear,NativeVariable},
    GenOpt.ExprTemplate{Int64,JuMP.VariableRef},
    GenOpt.ExprTemplate{Float64,JuMP.VariableRef},
}
const NativeIterator = Union{GenOpt.IteratorValues{Int64},GenOpt.IteratorValues{Float64}}
const NativeOperand = Union{NativeScalar,NativeTemplate,NativeIterator}
const NativeNode = Union{NativeOperand,Vector{NativeVariable},Vector{Float64}}
# Arrays for vector-valued functions and sets, and the arguments of nonlinear
# operators. They are not operands of jumpy_apply, which they would multiply
# the compiled operator combinations of.
const NativeVectorFunction = Union{Vector{NativeVariable},Vector{NativeAffine}}
const NativeArray = Union{
    Vector{NativeAffine},
    Matrix{NativeVariable},
    Vector{Int64},
    Matrix{Int64},
    Matrix{Float64},
}

mutable struct ModelHandle
    model::NativeModel
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
    value = (unsafe_pointer_to_objref(node)::Base.RefValue{Any})[]
    value isa NativeNode || error("Unsupported native expression type")
    return value::NativeNode
end

# Any native expression, including arrays.
function _unbox_value(node::Ptr{Cvoid})
    node == C_NULL && error("Expression pointer is NULL")
    value = (unsafe_pointer_to_objref(node)::Base.RefValue{Any})[]
    value isa Union{NativeNode,NativeArray} || error("Unsupported native expression type")
    return value::Union{NativeNode,NativeArray}
end

function _unbox_constraint(handle::ModelHandle, node::Ptr{Cvoid})
    node == C_NULL && error("Constraint pointer is NULL")
    any(ref -> pointer_from_objref(ref) == node, handle.nodes) ||
        error("Constraint does not belong to this model")
    value = (unsafe_pointer_to_objref(node)::Base.RefValue{Any})[]
    value isa MOI.ConstraintIndex || error(
        "Expected a scalar constraint; generated constraint groups have no scalar dual",
    )
    return value
end

include("native_dispatch.jl")

# Report without letting Julia exceptions cross the C ABI. Like
# `Base.showerror_nostdio`, but written to the file descriptor of stderr
# instead of its libuv stream: on Windows, the writes to that stream are queued
# and lost when Python exits the process without running the event loop.
function _print_error(@nospecialize(err))
    stderr_fd = Ptr{Cvoid}(2)
    ccall(:jl_printf, Cint, (Ptr{Cvoid}, Cstring), stderr_fd, "JuMPy error:\n")
    ccall(:jl_static_show, Csize_t, (Ptr{Cvoid}, Any), stderr_fd, err)
    ccall(:jl_printf, Cint, (Ptr{Cvoid}, Cstring), stderr_fd, "\n")
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

# -- Native sets --------------------------------------------------------------

Base.@ccallable function jumpy_abi_version()::Clonglong
    return JuMPy.ABI_VERSION
end

# No Julia object pointers leave this registry. A set can be reused across
# models in this image, but never in another backend's Julia runtime.
const NativeSet = Union{SET_TYPES...}
const NativeScalarSet = Union{filter(S -> S<:MOI.AbstractScalarSet, SET_TYPES)...}
const NativeVectorSet = Union{filter(S -> S<:MOI.AbstractVectorSet, SET_TYPES)...}
const SETS = Dict{UInt64,NativeSet}()
const SET_LOCK = ReentrantLock()
const NEXT_SET = Ref{UInt64}(1)

function _get_set(id::UInt64)
    Base.@lock SET_LOCK begin
        haskey(SETS, id) || throw(ArgumentError("Invalid or released native MOI set"))
        return SETS[id]
    end
end

function _store_set(set::NativeSet)
    Base.@lock SET_LOCK begin
        id = NEXT_SET[]
        id != 0 || error("Native MOI set handle space exhausted")
        SETS[id] = set
        NEXT_SET[] += UInt64(1)  # never reuse released handles
        return id
    end
end

# The sets of `SET_TYPES` as tab-separated lines: `set`, the 1-based index for
# jumpy_new_set, the name, the field names and the field types.
const REGISTRY = JuMPy.registry(SET_TYPES)
# The `JuMP.NonlinearOperator`s of the registry are applied by their heads,
# without dispatching on their functions, by jumpy_nonlinear.

# A NUL-terminated string, valid as long as the library is loaded.
Base.@ccallable function jumpy_registry()::Ptr{UInt8}
    return pointer(REGISTRY)
end

# Construct the set of the given registry index from the pointers to the data of
# its fields and their numbers of rows and columns, two per field.
Base.@ccallable function jumpy_new_set(
    index::Clonglong,
    args::Ptr{Ptr{Cvoid}},
    sizes::Ptr{Clonglong},
    nargs::Clonglong,
)::UInt64
    @_catch UInt64(0) begin
        set = JuMPy.new_set(Tuple{SET_TYPES...}, index, args, sizes, nargs)
        _store_set(set)
    end
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
        model = JuMPyDirect.model(Optimizer())
        handle = ModelHandle(model, Base.RefValue{Any}[])
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

# Returns the actual JuMP variable array as an opaque expression handle.
# Concrete and symbolic indexing both use jumpy_apply(:getindex, ...).
Base.@ccallable function jumpy_variables(model::Ptr{Cvoid}, count::Clonglong)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, JuMPyModel.add_variables(handle.model, count))
    end
end

# An array of variables of `ndims` dimensions `dims` constrained to belong to a
# vector set on creation, as `@variable(model, [1:dims[1], 1:dims[2]] in set)`.
Base.@ccallable function jumpy_constrained_variables(
    model::Ptr{Cvoid},
    dims::Ptr{Clonglong},
    ndims::Clonglong,
    set_id::UInt64,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        set = _get_set(set_id)
        set isa NativeVectorSet || error("Expected a vector set")
        variables = if ndims == 1
            size = (Int(unsafe_load(dims, 1)),)
            _dispatch_native(set, NativeVectorSet) do s
                JuMPyDirect.add_constrained_variables(handle.model, size, s)
            end
        elseif ndims == 2
            size = (Int(unsafe_load(dims, 1)), Int(unsafe_load(dims, 2)))
            _dispatch_native(set, NativeVectorSet) do s
                JuMPyDirect.add_constrained_variables(handle.model, size, s)
            end
        else
            error("Constrained variables have one or two dimensions")
        end
        _box(handle, variables)
    end
end

# -- Native expression constructors -------------------------------------------

Base.@ccallable function jumpy_constant(model::Ptr{Cvoid}, value::Cdouble)::Ptr{Cvoid}
    @_catch C_NULL _box(_get(model), value)
end

Base.@ccallable function jumpy_integer_constant(
    model::Ptr{Cvoid},
    value::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL _box(_get(model), value)
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
        head == C_NULL && error("Operator name is NULL")
        op = Symbol(unsafe_string(head))
        args == C_NULL && error("Operator arguments are NULL")
        func = if nargs == 1
            _dispatch_native(_unbox(unsafe_load(args, 1)), NativeNode) do arg
                _apply_native(op, arg)
            end
        elseif nargs == 2
            _dispatch_native(
                _unbox(unsafe_load(args, 1)),
                _unbox(unsafe_load(args, 2)),
                NativeNode,
                NativeNode,
            ) do left, right
                _apply_native(op, left, right)
            end
        else
            error("Native operators require one or two arguments")
        end
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
        _box(handle, JuMPyModel.iterator([unsafe_load(values, k) for k = 1:len]))
    end
end

Base.@ccallable function jumpy_integer_iterator(
    model::Ptr{Cvoid},
    values::Ptr{Clonglong},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, JuMPyModel.iterator([unsafe_load(values, k) for k = 1:len]))
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
        _box(handle, [unsafe_load(values, k) for k = 1:len])
    end
end

function _data_array(::Type{T}, data::Ptr{Cvoid}, rows, columns, ndims) where {T}
    ptr = Ptr{T}(data)
    ndims == 1 && return T[unsafe_load(ptr, k) for k = 1:rows]
    ndims == 2 &&
        return T[unsafe_load(ptr, r + (c - 1) * rows) for r = 1:rows, c = 1:columns]
    return error("Data arrays have one or two dimensions")
end

# A data array of `ndims` dimensions `rows` and `columns`, of Int64 elements if
# `integer` is nonzero and Float64 elements otherwise. Matrices are
# column-major.
Base.@ccallable function jumpy_array(
    model::Ptr{Cvoid},
    integer::Cint,
    data::Ptr{Cvoid},
    rows::Clonglong,
    columns::Clonglong,
    ndims::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        array = if integer != 0
            _data_array(Int64, data, rows, columns, ndims)
        else
            _data_array(Float64, data, rows, columns, ndims)
        end
        _box(handle, array)
    end
end

# A 1-based index of jumpy_getindex, as a range and whether it is the integer
# `first`, when `last` is typemin(Int64), rather than the range `first:last`.
# Indexing with ranges only is compiled once per array type.
function _range(first::Clonglong, last::Clonglong)
    scalar = last == typemin(Clonglong)
    return Int64(first):Int64(scalar ? first : last), scalar
end

const NativeIndexable = Union{
    Vector{NativeVariable},
    Matrix{NativeVariable},
    Vector{NativeAffine},
    Vector{Float64},
    Vector{Int64},
    Matrix{Float64},
    Matrix{Int64},
}

# Index an array with 1-based integers or ranges, one per dimension, given by
# `first` and `last` (see `_range`): `x[i, j]` is a variable while `x[1:m, j]`
# is a vector, as in Julia.
Base.@ccallable function jumpy_getindex(
    model::Ptr{Cvoid},
    array::Ptr{Cvoid},
    first::Ptr{Clonglong},
    last::Ptr{Clonglong},
    nindices::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        value = _unbox_value(array)
        value isa NativeIndexable || error("Expected a native array")
        rows, row = _range(unsafe_load(first, 1), unsafe_load(last, 1))
        result = if nindices == 1
            _dispatch_native(value, NativeIndexable) do a
                ndims(a) == 1 || error("Expected $(ndims(a)) indices")
                entries = a[rows]
                row ? entries[1] : entries
            end
        elseif nindices == 2
            columns, column = _range(unsafe_load(first, 2), unsafe_load(last, 2))
            _dispatch_native(value, NativeIndexable) do a
                ndims(a) == 2 || error("Expected $(ndims(a)) indices")
                entries = a[rows, columns]
                row && column ? entries[1, 1] : row || column ? vec(entries) : entries
            end
        else
            error("Arrays have one or two dimensions")
        end
        _box(handle, result)
    end
end

# A single method for the untyped values of jumpy_vcat, so that its call is
# resolved statically.
function _append_affine!(result::Vector{NativeAffine}, @nospecialize(value))
    if value isa Union{Int64,Float64,NativeVariable,NativeAffine}
        push!(result, convert(NativeAffine, value))
    elseif value isa NativeVectorFunction
        for x in value
            push!(result, convert(NativeAffine, x))
        end
    else
        error("Expected numbers, variables, affine expressions or vectors of them")
    end
    return result
end

# The vector of the affine expressions `args`, where vectors are concatenated,
# as `vcat` and `[a; b]` in Julia.
Base.@ccallable function jumpy_vcat(
    model::Ptr{Cvoid},
    args::Ptr{Ptr{Cvoid}},
    nargs::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        result = NativeAffine[]
        for k = 1:nargs
            _append_affine!(result, _unbox_value(unsafe_load(args, k)))
        end
        _box(handle, result)
    end
end

# The nonlinear expression `head(args...)`, such as a call of a
# `JuMP.NonlinearOperator` of the registry with this head.
Base.@ccallable function jumpy_nonlinear(
    model::Ptr{Cvoid},
    head::Cstring,
    args::Ptr{Ptr{Cvoid}},
    nargs::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        head == C_NULL && error("Operator name is NULL")
        values = Any[_unbox_value(unsafe_load(args, k)) for k = 1:nargs]
        _box(handle, NativeNonlinear(Symbol(unsafe_string(head)), values))
    end
end

# -- Constraints --------------------------------------------------------------

# JuMP builds the constraints; the compiled adapter sends them to raw MOI.
# The actual constraint index (or generated vector) stays rooted in its model.
Base.@ccallable function jumpy_add_constraint(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
    set_id::UInt64,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        set = _get_set(set_id)
        handle = _get(model)
        expression = _unbox_value(func)
        if expression isa NativeVectorFunction
            set isa NativeVectorSet || error("Expected a vector set")
            constraint = _dispatch_native(
                expression,
                set,
                NativeVectorFunction,
                NativeVectorSet,
            ) do f, s
                JuMPyDirect.add_constraint(handle.model, f, s)
            end
            return _box(handle, constraint)
        end
        expression isa Union{NativeScalar,NativeTemplate} ||
            error("Expected a scalar or generated constraint expression")
        set isa NativeScalarSet || error("Expected a scalar set")
        constraint = _dispatch_native(
            expression,
            set,
            Union{NativeScalar,NativeTemplate},
            NativeScalarSet,
        ) do f, s
            JuMPyDirect.add_constraint(handle.model, f, s)
        end
        _box(handle, constraint)
    end
end

# -- Objective ----------------------------------------------------------------

# MOI.set(MOI.ObjectiveSense()); 0 = min, 1 = max.
Base.@ccallable function jumpy_set_objective_sense(model::Ptr{Cvoid}, sense::Cint)::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        moi_sense = sense == 0 ? MOI.MIN_SENSE : MOI.MAX_SENSE
        JuMPyDirect.set_objective_sense(handle.model, moi_sense)
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
        expression = _unbox(func)
        expression isa NativeScalar || error("Expected a scalar objective expression")
        _dispatch_native(expression, NativeScalar) do f
            JuMPyDirect.set_objective_function(handle.model, f)
        end
        Cint(0)
    end
end

# -- Solve and solution retrieval ----------------------------------------------

Base.@ccallable function jumpy_set_time_limit_sec(model::Ptr{Cvoid}, seconds::Cdouble)::Cint
    @_catch Cint(-1) begin
        JuMPyModel.set_time_limit_sec(_get(model).model, seconds)
        Cint(0)
    end
end

# JuMP.set_attribute(model, name, value), with one entry point per value type.
function _set_attribute(model::Ptr{Cvoid}, name::Cstring, value)
    JuMPyModel.set_attribute(_get(model).model, unsafe_string(name), value)
    return Cint(0)
end

Base.@ccallable function jumpy_set_string_attribute(
    model::Ptr{Cvoid},
    name::Cstring,
    value::Cstring,
)::Cint
    @_catch Cint(-1) _set_attribute(model, name, unsafe_string(value))
end

Base.@ccallable function jumpy_set_bool_attribute(
    model::Ptr{Cvoid},
    name::Cstring,
    value::Cint,
)::Cint
    @_catch Cint(-1) _set_attribute(model, name, value != 0)
end

Base.@ccallable function jumpy_set_int_attribute(
    model::Ptr{Cvoid},
    name::Cstring,
    value::Clonglong,
)::Cint
    @_catch Cint(-1) _set_attribute(model, name, Int64(value))
end

Base.@ccallable function jumpy_set_float_attribute(
    model::Ptr{Cvoid},
    name::Cstring,
    value::Cdouble,
)::Cint
    @_catch Cint(-1) _set_attribute(model, name, Float64(value))
end

# Returns Int(MOI.TerminationStatusCode); MOI.OPTIMAL is 1.
Base.@ccallable function jumpy_optimize(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        JuMPyDirect.optimize(handle.model)
        Cint(Integer(MOI.get(JuMP.backend(handle.model), MOI.TerminationStatus())))
    end
end

# Read the solution through the actual JuMP variable or expression.
Base.@ccallable function jumpy_value(model::Ptr{Cvoid}, func::Ptr{Cvoid})::Cdouble
    @_catch Cdouble(NaN) begin
        _get(model)
        expression = _unbox(func)
        expression isa NativeScalar || error("Expected a scalar value expression")
        _dispatch_native(expression, NativeScalar) do f
            Cdouble(JuMPyDirect.value(f))
        end
    end
end

Base.@ccallable function jumpy_objective_value(model::Ptr{Cvoid})::Cdouble
    @_catch Cdouble(NaN) Cdouble(JuMPyModel.objective_value(_get(model).model))
end

Base.@ccallable function jumpy_has_values(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) Cint(JuMPyModel.has_values(_get(model).model))
end

Base.@ccallable function jumpy_has_duals(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) Cint(JuMPyModel.has_duals(_get(model).model))
end

Base.@ccallable function jumpy_termination_status(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) Cint(Integer(JuMPyModel.termination_status(_get(model).model)))
end

Base.@ccallable function jumpy_dual(model::Ptr{Cvoid}, constraint::Ptr{Cvoid})::Cdouble
    @_catch Cdouble(NaN) begin
        handle = _get(model)
        index = _unbox_constraint(handle, constraint)
        Cdouble(JuMPyDirect.dual(handle.model, index))
    end
end
