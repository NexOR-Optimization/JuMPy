# Construct a set of a backend from the C arguments of `jumpy_new_set`: the
# pointers `args` to the data of its fields, and their numbers of rows and
# columns `sizes`, two per field.

_load(::Type{T}, ptr::Ptr{Cvoid}, k) where {T<:Union{Int64,Float64}} =
    unsafe_load(Ptr{T}(ptr), k)
# Python passes booleans as bytes.
_load(::Type{Bool}, ptr::Ptr{Cvoid}, k) = unsafe_load(Ptr{UInt8}(ptr), k) != 0

function _size(sizes::Ptr{Clonglong}, i::Int)
    return Int(unsafe_load(sizes, 2i - 1)), Int(unsafe_load(sizes, 2i))
end

function _argument(::Type{T}, args, sizes, i::Int) where {T<:Union{Int64,Float64,Bool}}
    _size(sizes, i) == (1, 1) || throw(DimensionMismatch("Expected a scalar argument $i"))
    return _load(T, unsafe_load(args, i), 1)
end

function _argument(::Type{Vector{T}}, args, sizes, i::Int) where {T}
    rows, columns = _size(sizes, i)
    columns == 1 || throw(DimensionMismatch("Expected a vector argument $i"))
    ptr = unsafe_load(args, i)
    return T[_load(T, ptr, k) for k = 1:rows]
end

function _argument(::Type{Matrix{T}}, args, sizes, i::Int) where {T}
    rows, columns = _size(sizes, i)
    ptr = unsafe_load(args, i)
    return T[_load(T, ptr, r + (c - 1) * rows) for r = 1:rows, c = 1:columns]
end

# The set `S` from its fields. `S` is a parameter of the method so that it is
# compiled for each set type. The generator cannot call methods defined after
# JuMPy, such as the `show` of the enum parameters of a set.
@generated function _construct(::Type{S}, args, sizes, nargs) where {S}
    n = fieldcount(S)
    fields = [:(_argument($(fieldtype(S, i)), args, sizes, $i)) for i = 1:n]
    message = "$(nameof(S)) expects $n arguments"
    return quote
        nargs == $n || throw(ArgumentError($message))
        return $S($(fields...))
    end
end

"""
    new_set(::Type{Tuple{S1,S2,...}}, index, args, sizes, nargs)

Construct the set type of the given (1-based) `index` among `S1`, `S2`, ...,
from its fields. See `jumpy_new_set`.
"""
@generated function new_set(::Type{T}, index, args, sizes, nargs) where {T<:Tuple}
    body = :(throw(ArgumentError("Invalid native set index")))
    for (k, S) in reverse(collect(enumerate(T.parameters)))
        body = :(index == $k ? _construct($S, args, sizes, nargs) : $body)
    end
    return body
end
