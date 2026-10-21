module ConcreteArithmeticProbe

import JuMP
import MathOptInterface as MOI

const ROOTS = Base.RefValue{Any}[]
const MODEL = JuMP.concrete_direct_model(MOI.Utilities.Model{Float64}())
const X = JuMP.@variable(MODEL)
const NativeVariable = typeof(X)
const NativeAffine = typeof(X + 0.0)
const NativeQuadratic = typeof(X * X)
const NativeScalarTypes = (Int64, Float64, NativeVariable, NativeAffine)
push!(ROOTS, Ref{Any}(X), Ref{Any}(2), Ref{Any}(1.0))

Base.@ccallable function probe_node(index::Cint)::Ptr{Cvoid}
    return pointer_from_objref(ROOTS[index])
end

macro native_binary(op, a, b)
    branches = Expr[]
    for f in (:+, :*)
        body = :(throw(ArgumentError("Unsupported native operand types")))
        for A in reverse(NativeScalarTypes), B in reverse(NativeScalarTypes)
            call = Expr(:call, GlobalRef(Base, f), :($a::$A), :($b::$B))
            body = :(if $a isa $A && $b isa $B; $call; else; $body; end)
        end
        push!(branches, body)
    end
    return esc(quote
        if $op === :+
            $(branches[1])
        elseif $op === :*
            $(branches[2])
        else
            throw(ArgumentError("Unsupported native operation"))
        end
    end)
end

Base.@ccallable function probe_apply(
    op::Cstring, left::Ptr{Cvoid}, right::Ptr{Cvoid},
)::Ptr{Cvoid}
    try
        a = (unsafe_pointer_to_objref(left)::Base.RefValue{Any})[]
        b = (unsafe_pointer_to_objref(right)::Base.RefValue{Any})[]
        operation = Symbol(unsafe_string(op))
        result = @native_binary(operation, a, b)
        boxed = Base.RefValue{Any}(result)
        push!(ROOTS, boxed)
        return pointer_from_objref(boxed)
    catch err
        Base.showerror_nostdio(err, "ConcreteArithmeticProbe error")
        return C_NULL
    end
end

Base.@ccallable function probe_evaluate(ref::Ptr{Cvoid})::Cdouble
    try
        value = (unsafe_pointer_to_objref(ref)::Base.RefValue{Any})[]
        if value isa Int64 || value isa Float64
            return Float64(value)
        elseif value isa NativeVariable
            return 3.0
        elseif value isa NativeAffine
            return JuMP.value(_ -> 3.0, value)
        elseif value isa NativeQuadratic
            return JuMP.value(_ -> 3.0, value)
        end
        throw(ArgumentError("Unsupported native value type"))
    catch err
        Base.showerror_nostdio(err, "ConcreteArithmeticProbe value error")
        return NaN
    end
end

end
