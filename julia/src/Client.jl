"""
    Client

Call the C entry points of a backend module `J`, which called
[`JuMPy.@entrypoints`](@ref), as the `jumpy` Python package does. This is used
by the workloads traced before building an image, and by tests.
"""
module Client

# The 1-based registry index of the set named `name`, and its field types.
function _registry_entry(J::Module, name::String)
    for line in eachline(IOBuffer(unsafe_string(J.jumpy_registry())))
        kind, index, set_name, _, types = split(line, '\t')
        if kind == "set" && set_name == name
            return parse(Int, index), isempty(types) ? String[] : split(types, ';')
        end
    end
    return error("No set $name in the registry of $J")
end

# The C data of a set field and its numbers of rows and columns. Matrices are
# column-major, as in Julia.
_data(::Type{T}, value) where {T} = T[value], (1, 1)
_data(::Type{T}, value::AbstractVector) where {T} = Vector{T}(value), (length(value), 1)
_data(::Type{T}, value::AbstractMatrix) where {T} = Matrix{T}(value), size(value)

_element_type(code) = Dict("Int64" => Int64, "Float64" => Float64, "Bool" => UInt8)[replace(
    code,
    r"^(?:Vector|Matrix)\{(.*)\}$" => s"\1",
)]

"""
    new_set(J, name, args...)

Construct the set `name` of the registry, such as
`"MathOptInterface.LessThan"`, with the field values `args`, through
`jumpy_new_set`. Return its handle, which is zero on error.
"""
function new_set(J::Module, name::String, args...)
    index, types = _registry_entry(J, name)
    length(args) == length(types) ||
        error("$name expects $(length(types)) arguments, got $(length(args))")
    data = [_data(_element_type(code), arg) for (code, arg) in zip(types, args)]
    arrays = first.(data)
    pointers = Ptr{Cvoid}[pointer(a) for a in arrays]
    sizes = Clonglong[n for (_, s) in data for n in s]
    return GC.@preserve arrays pointers sizes J.jumpy_new_set(
        Clonglong(index),
        pointer(pointers),
        pointer(sizes),
        Clonglong(length(arrays)),
    )
end

"""
    apply(J, model, head, args::Ptr{Cvoid}...)

Apply the Julia operator `head` to the native expressions `args`.
"""
function apply(J::Module, model, head::String, args::Ptr{Cvoid}...)
    argv = collect(Ptr{Cvoid}, args)
    return GC.@preserve head argv J.jumpy_apply(
        model,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

constant(J::Module, model, value::Real) = J.jumpy_constant(model, Cdouble(value))
constant(J::Module, model, value::Integer) =
    J.jumpy_integer_constant(model, Clonglong(value))

"""
    constrained_variables(J, model, dims, set)

The array of variables of dimensions `dims`, such as `(m, n)`, constrained to
belong to `set`, a handle returned by [`new_set`](@ref).
"""
function constrained_variables(J::Module, model, dims::Tuple, set::UInt64)
    sizes = Clonglong[dims...]
    return GC.@preserve sizes J.jumpy_constrained_variables(
        model,
        pointer(sizes),
        Clonglong(length(sizes)),
        set,
    )
end

"""
    index(J, model, array, indices...)

Index `array` with 1-based integers or `UnitRange`s, one per dimension.
"""
function index(J::Module, model, array, indices::Union{Integer,UnitRange}...)
    first = Clonglong[i isa Integer ? i : Base.first(i) for i in indices]
    last = Clonglong[i isa Integer ? typemin(Clonglong) : Base.last(i) for i in indices]
    return GC.@preserve first last J.jumpy_getindex(
        model,
        array,
        pointer(first),
        pointer(last),
        Clonglong(length(indices)),
    )
end

"""
    concatenate(J, model, args::Ptr{Cvoid}...)

The vector of the affine expressions `args`, where vectors are concatenated.
"""
function concatenate(J::Module, model, args::Ptr{Cvoid}...)
    argv = collect(Ptr{Cvoid}, args)
    return GC.@preserve argv J.jumpy_vcat(model, pointer(argv), Clonglong(length(argv)))
end

"""
    array(J, model, values::Array)

The data vector or matrix `values` of `Int64` or `Float64` elements.
"""
function array(J::Module, model, values::Array{T}) where {T<:Union{Int64,Float64}}
    rows, columns = size(values, 1), size(values, 2)
    return GC.@preserve values J.jumpy_array(
        model,
        Cint(T == Int64),
        Ptr{Cvoid}(pointer(values)),
        Clonglong(rows),
        Clonglong(columns),
        Clonglong(ndims(values)),
    )
end

"""
    nonlinear(J, model, head, args::Ptr{Cvoid}...)

The nonlinear expression `head(args...)`, such as the call of a
`JuMP.NonlinearOperator` of this head.
"""
function nonlinear(J::Module, model, head::String, args::Ptr{Cvoid}...)
    argv = collect(Ptr{Cvoid}, args)
    return GC.@preserve head argv J.jumpy_nonlinear(
        model,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

"""
    variable(J, model, block, i)

The variable of 1-based index `i` of the variable array `block`.
"""
variable(J::Module, model, block, i::Integer) =
    apply(J, model, "getindex", block, constant(J, model, i))

"""
    add_constraint(J, model, func, set)

Add the constraint that the native expression `func` belongs to the set
`set`, a handle returned by [`new_set`](@ref). The handle is not released.
"""
add_constraint(J::Module, model, func, set::UInt64) =
    J.jumpy_add_constraint(model, func, set)

"""
    add_constraint(J, model, func, name, args...)

Add the constraint that `func` belongs to the set `name` of the registry
with fields `args`, releasing the set afterwards.
"""
function add_constraint(J::Module, model, func, name::String, args...)
    set = new_set(J, name, args...)
    try
        return add_constraint(J, model, func, set)
    finally
        J.jumpy_free_set(set)
    end
end

function set_objective(J::Module, model, sense::Integer, func)
    return J.jumpy_set_objective_sense(model, Cint(sense)) == 0 &&
           J.jumpy_set_objective_function(model, func) == 0
end

function set_attribute(J::Module, model, name::String, value)
    GC.@preserve name value begin
        key = Cstring(pointer(name))
        if value isa String
            return J.jumpy_set_string_attribute(model, key, Cstring(pointer(value)))
        elseif value isa Bool
            return J.jumpy_set_bool_attribute(model, key, Cint(value))
        elseif value isa Integer
            return J.jumpy_set_int_attribute(model, key, Clonglong(value))
        else
            return J.jumpy_set_float_attribute(model, key, Cdouble(value))
        end
    end
end

"""
    scalar_sets(J)

The names of the sets of the registry of `J` whose fields are scalars, with a
value of their fields for tests and workloads.
"""
function scalar_sets(J::Module)
    sets = Tuple{String,Vector{Any}}[]
    for line in eachline(IOBuffer(unsafe_string(J.jumpy_registry())))
        kind, _, name, _, types = split(line, '\t')
        kind == "set" || continue
        fields = isempty(types) ? String[] : split(types, ';')
        all(in(("Int64", "Float64", "Bool")), fields) || continue
        push!(sets, (String(name), Any[_sample(field) for field in fields]))
    end
    return sets
end

_sample(field) = Dict("Int64" => 1, "Float64" => 1.0, "Bool" => true)[field]

"""
    workload(J; attributes = (), optimize = J.jumpy_optimize)

Exercise the C ABI of the backend module `J` as Python does, to trace the
dynamic dispatches of an ordinary Julia run before building its image. The
pairs `attributes` are set as optimizer attributes, and `optimize(model)`
solves the model.
"""
function workload(J::Module; attributes = (), optimize = J.jumpy_optimize)
    m = J.jumpy_new_model()
    for (name, value) in attributes
        @assert set_attribute(J, m, name, value) == 0
    end
    @assert J.jumpy_set_time_limit_sec(m, 10.0) == 0
    @assert J.jumpy_set_time_limit_sec(m, NaN) == 0
    variables = J.jumpy_variables(m, Clonglong(3))
    x = variable(J, m, variables, 1)
    # Every scalar set, and the variable bounds and integrality of Python.
    for (name, args) in scalar_sets(J)
        add_constraint(J, m, variable(J, m, variables, 3), name, args...)
    end
    add_constraint(J, m, x, "MathOptInterface.GreaterThan", 0.0)
    add_constraint(J, m, x, "MathOptInterface.LessThan", 10.0)
    add_constraint(J, m, variable(J, m, variables, 2), "MathOptInterface.ZeroOne")
    # Affine expressions with every operator Python emits.
    one = constant(J, m, 1.0)
    plus = apply(J, m, "+", x, one)
    add_constraint(J, m, plus, "MathOptInterface.LessThan", 5.0)
    add_constraint(J, m, plus, "MathOptInterface.GreaterThan", 0.0)
    add_constraint(J, m, plus, "MathOptInterface.EqualTo", 3.0)
    scaled = apply(J, m, "*", constant(J, m, 2.0), apply(J, m, "-", x, one))
    divided = apply(J, m, "/", scaled, constant(J, m, 4.0))
    add_constraint(J, m, divided, "MathOptInterface.LessThan", 8.0)
    # Groups: iterator, variable block, data array, 1-D and 2-D. GenOpt
    # caches the values of an iterator, so each group gets a fresh iterator
    # to trace the first expansion.
    block = J.jumpy_variables(m, Clonglong(3))
    function iterator()
        values = Clonglong[0, 1, 2]
        i = GC.@preserve values J.jumpy_integer_iterator(m, pointer(values), Clonglong(3))
        return apply(J, m, "+", i, constant(J, m, 1))
    end
    data = Cdouble[1.0, 2.0, 3.0]
    d = GC.@preserve data J.jumpy_float_array(m, pointer(data), Clonglong(3))
    i1 = iterator()
    entry = apply(J, m, "getindex", block, i1)
    template = apply(J, m, "-", entry, apply(J, m, "getindex", d, i1))
    add_constraint(J, m, template, "MathOptInterface.GreaterThan", 0.0)
    add_constraint(J, m, entry, "MathOptInterface.GreaterThan", 1.0)
    pair = Cdouble[0.0, 1.0]
    j = GC.@preserve pair J.jumpy_iterator(m, pointer(pair), Clonglong(2))
    template2 = apply(J, m, "-", apply(J, m, "+", entry, j), one)
    add_constraint(J, m, template2, "MathOptInterface.LessThan", 5.0)
    # Objective: both senses, function and bare-variable forms.
    set_objective(J, m, 0, plus)
    set_objective(J, m, 1, x)
    optimize(m)
    J.jumpy_termination_status(m)
    J.jumpy_has_values(m)
    J.jumpy_has_duals(m)
    J.jumpy_objective_value(m)
    J.jumpy_value(m, x)
    J.jumpy_value(m, plus)
    # Nonlinear expressions can be evaluated at the solution even when the
    # solver does not accept nonlinear constraints or objectives.
    for head in ("sin", "cos", "exp", "log", "sqrt", "abs")
        nonlinear = apply(J, m, head, x)
        J.jumpy_value(m, apply(J, m, "+", nonlinear, apply(J, m, head, plus)))
    end
    # Every remaining operator Python can emit, including those that end in
    # the unsupported-constraint error path, which is also worth exercising.
    add_constraint(J, m, apply(J, m, "-", x), "MathOptInterface.LessThan", 1.0)
    square = apply(J, m, "^", x, constant(J, m, 2.0))
    add_constraint(J, m, square, "MathOptInterface.LessThan", 9.0)
    for head in ("sin", "cos", "exp", "log", "sqrt", "abs")
        add_constraint(J, m, apply(J, m, head, x), "MathOptInterface.LessThan", 1.0)
        generated = apply(J, m, head, apply(J, m, "getindex", block, iterator()))
        add_constraint(J, m, generated, "MathOptInterface.LessThan", 1.0)
    end
    # Lazy sums, in constraints and objectives, and quadratic groups.
    lazy = apply(J, m, "lazy_sum", apply(J, m, "getindex", block, iterator()))
    add_constraint(J, m, lazy, "MathOptInterface.LessThan", 9.0)
    k = iterator()
    weighted = apply(J, m, "*", k, apply(J, m, "getindex", block, k))
    set_objective(J, m, 1, apply(J, m, "lazy_sum", weighted))
    k = iterator()
    cost =
        apply(J, m, "*", apply(J, m, "getindex", d, k), apply(J, m, "getindex", block, k))
    set_objective(J, m, 0, apply(J, m, "lazy_sum", cost))
    optimize(m)
    k = iterator()
    gap = apply(J, m, "-", apply(J, m, "getindex", block, k), k)
    square = apply(J, m, "^", gap, constant(J, m, 2))
    set_objective(J, m, 0, apply(J, m, "lazy_sum", square))
    optimize(m)
    entry = apply(J, m, "getindex", block, iterator())
    squares = apply(J, m, "*", entry, entry)
    add_constraint(J, m, squares, "MathOptInterface.LessThan", 4.0)
    # Errors remain contained within the C entry points.
    J.jumpy_optimize(C_NULL)
    J.jumpy_free_model(m)
    J.jumpy_free_set(UInt64(0))
    J.jumpy_add_constraint(C_NULL, C_NULL, UInt64(0))
    return
end

end # module Client
