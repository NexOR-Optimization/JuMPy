# The sets that a compiled backend can construct from Python, with their names
# and the C types of their fields. They are found when the backend package is
# precompiled: reflection is not available in the trimmed image.

# The types of set fields that cross the C ABI, with their codes in the
# registry. Arrays are passed as a data pointer with their number of rows and
# columns; matrices are column-major.
const ARGUMENT_TYPES = (
    Int64 => "Int64",
    Float64 => "Float64",
    Bool => "Bool",
    Vector{Int64} => "Vector{Int64}",
    Vector{Float64} => "Vector{Float64}",
    Vector{Bool} => "Vector{Bool}",
    Matrix{Int64} => "Matrix{Int64}",
    Matrix{Float64} => "Matrix{Float64}",
    Matrix{Bool} => "Matrix{Bool}",
)

_argument_code(T) = something(findfirst(p -> p.first === T, ARGUMENT_TYPES), 0)

"""
    constructible(S::Type)

Whether Python can construct the set `S` from its fields, that is, whether `S`
is concrete and each of its field types is one of `ARGUMENT_TYPES`.
"""
function constructible(S::Type)
    isconcretetype(S) && S <: MOI.AbstractSet || return false
    return all(i -> _argument_code(fieldtype(S, i)) != 0, 1:fieldcount(S))
end

# Make a set type concrete by choosing `Float64` for its free numeric
# parameters, such as `T` in `MOI.LessThan{T}`. Parameters that are values,
# such as `W` in `MathOptVRP.TimeWindows{W}`, must already be given.
function _concrete(S)
    while S isa UnionAll
        var = S.var
        Float64 <: var.ub && var.lb <: Float64 || return nothing
        S = S{Float64}
    end
    return S
end

# The set types `X` of a signature argument `Type{X}`, `Type{<:X}` or
# `Type{S} where {S<:X}`, with `X` possibly a union.
function _signature_sets(T)
    T = Base.unwrap_unionall(T)
    T isa DataType && T <: Type || return Any[]
    X = only(T.parameters)
    if X isa TypeVar
        X = X.ub
    end
    return Any[Base.uniontypes(X)...]
end

"""
    supported_sets(optimizer_type::Type)

The constructible set types that `optimizer_type` declares in its own
`MOI.supports_constraint`, `MOI.supports_add_constrained_variable` and
`MOI.supports_add_constrained_variables` methods. Methods of supertypes, such
as the `false` fallbacks for any `MOI.ModelLike`, are not considered.
"""
function supported_sets(optimizer_type::Type)
    sets = Any[]
    for (f, nargs) in (
        (MOI.supports_constraint, 3),
        (MOI.supports_add_constrained_variable, 2),
        (MOI.supports_add_constrained_variables, 2),
    )
        for method in methods(f)
            signature = Base.unwrap_unionall(method.sig)
            parameters = signature.parameters
            length(parameters) == nargs + 1 || continue
            O = parameters[2]
            O isa Type && O <: MOI.AbstractOptimizer && optimizer_type <: O || continue
            # Exclude the generic methods, which define no specific support.
            O in (MOI.AbstractOptimizer, MOI.ModelLike) && continue
            for S in _signature_sets(parameters[end])
                S = _concrete(S)
                if S !== nothing && constructible(S) && !(S in sets)
                    push!(sets, S)
                end
            end
        end
    end
    return Tuple(sort!(sets; by = set_name))
end

# The module and name of `S` for Python, followed by its value parameters, if
# any. Type parameters are omitted: `MOI.LessThan{Float64}` is
# `MathOptInterface.LessThan` while `TimeWindows{WITHOUT_START_TIME,Float64}`
# is `MathOptVRP.TimeWindows{WITHOUT_START_TIME}`.
function set_name(S::Type)
    name = string(nameof(parentmodule(S)), ".", nameof(S))
    values = [p for p in S.parameters if !(p isa Type)]
    if !isempty(values)
        name *= "{" * join(string.(values), ",") * "}"
    end
    return name
end

"""
    operators(set_types)

The names and heads of the `JuMP.NonlinearOperator`s defined in the modules of
`set_types` other than MathOptInterface, such as
`MathOptVRP.op_sum_distances`.
"""
function operators(set_types::Tuple)
    modules = unique!([parentmodule(S) for S in set_types])
    result = Tuple{String,Symbol}[]
    for M in modules
        M === MOI && continue
        for name in names(M; all = true)
            startswith(string(name), "#") && continue
            isdefined(M, name) || continue
            value = getfield(M, name)
            if value isa JuMP.NonlinearOperator
                push!(result, (string(nameof(M), ".", name), value.head))
            end
        end
    end
    return result
end

"""
    registry(set_types)

The tab-separated description of the sets and operators returned by
`jumpy_registry`: one line `set`, index, name, field names and field types
per set, and one line `operator`, index, name and head per operator of
[`operators`](@ref).
"""
function registry(set_types::Tuple)
    io = IOBuffer()
    for (index, S) in enumerate(set_types)
        names = join(string.(fieldnames(S)), ",")
        types = join(
            (
                ARGUMENT_TYPES[_argument_code(fieldtype(S, i))].second for
                i = 1:fieldcount(S)
            ),
            ";",
        )
        println(io, "set\t", index, "\t", set_name(S), "\t", names, "\t", types)
    end
    for (index, (name, head)) in enumerate(operators(set_types))
        println(io, "operator\t", index, "\t", name, "\t", head, "\t")
    end
    return String(take!(io))
end
