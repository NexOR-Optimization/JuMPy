# Explicit experimental MOI specialization, not loaded by the normal backend.
# Preserve the existing plain-text VariableIndex representation without the
# general AbstractFunction display path rejected by the trim verifier.
import MathOptInterface as MOI

function Base.show(io::IO, x::MOI.VariableIndex)
    print(io, "MOI.VariableIndex(", x.value, ")")
    return
end

function Base.show(io::IO, ::MIME"text/plain", x::MOI.VariableIndex)
    return show(io, x)
end
