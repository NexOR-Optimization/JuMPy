# Recover the finite operator/type combinations supported by this compiled
# image. These are the ordinary JuMP/GenOpt objects and Base operators; Python
# continues to pass opaque handles without classifying expressions.

# A union annotation alone leaves large combinations to runtime dispatch.
# Split at the handle boundary so every call specializes on a known type.
@generated function _dispatch_native(f::F, value, ::Type{U}) where {F,U}
    body = :(error("Unsupported native expression type"))
    for T in reverse(Base.uniontypes(U))
        body = :(
            if value isa $T
                f(value::$T)
            else
                $body
            end
        )
    end
    return body
end

@generated function _dispatch_native(f::F, left, right, ::Type{L}, ::Type{R}) where {F,L,R}
    body = :(error("Unsupported native expression type"))
    for A in reverse(Base.uniontypes(L))
        right_body = :(error("Unsupported native expression type"))
        for B in reverse(Base.uniontypes(R))
            right_body = :(
                if right isa $B
                    f(left::$A, right::$B)
                else
                    $right_body
                end
            )
        end
        body = :(
            if left isa $A
                $right_body
            else
                $body
            end
        )
    end
    # Keep the checked branches at the C entry point: safe verification can
    # otherwise leave this dispatcher call unresolved for large type unions.
    return Expr(:block, Expr(:meta, :inline), body)
end

function _apply_native(op::Symbol, arg::NativeNode)
    arg isa NativeOperand || error("Expected a scalar or generated operand")
    op === :+ && return +arg
    op === :- && return -arg
    op === :sin && return sin(arg)
    op === :cos && return cos(arg)
    op === :exp && return exp(arg)
    op === :log && return log(arg)
    op === :sqrt && return sqrt(arg)
    op === :abs && return abs(arg)
    return error("Unsupported native unary operator")
end

function _apply_native(op::Symbol, left::NativeNode, right::NativeNode)
    if op === :getindex
        left isa Union{Vector{NativeVariable},Vector{Float64}} ||
            error("Expected a native array")
        right isa Union{Int64,NativeTemplate,NativeIterator} ||
            error("Expected an integer or generated index")
        return getindex(left, right)
    end
    left isa NativeOperand && right isa NativeOperand ||
        error("Expected scalar or generated operands")
    op === :+ && return left + right
    op === :- && return left - right
    op === :* && return left * right
    op === :/ && return left / right
    op === :^ && return left ^ right
    return error("Unsupported native binary operator")
end
