module JuMPyMOI

# Shared by JuliaCall (loaded from the Python package) and the JuliaC image.
# This module owns no optimizer, runtime, or opaque-pointer registry. All
# values are ordinary MOI objects in the runtime that loads this source.
import MathOptInterface as MOI

function scalar_nonlinear(head::Symbol, args::Vector{Any})
    return MOI.ScalarNonlinearFunction(head, args)
end

# These tags are the existing jumpy_add_constraint C ABI, not new set types.
# Val methods let the compiled backend construct each concrete set without
# dynamic dispatch through a union of all five possible return types.
scalar_set(::Val{0}, rhs::Float64) = MOI.LessThan(rhs)
scalar_set(::Val{1}, rhs::Float64) = MOI.GreaterThan(rhs)
scalar_set(::Val{2}, rhs::Float64) = MOI.EqualTo(rhs)
scalar_set(::Val{3}, ::Float64) = MOI.ZeroOne()
scalar_set(::Val{4}, ::Float64) = MOI.Integer()

function scalar_set(sense::Integer, rhs::Float64)
    if sense == 0
        return scalar_set(Val(0), rhs)
    elseif sense == 1
        return scalar_set(Val(1), rhs)
    elseif sense == 2
        return scalar_set(Val(2), rhs)
    elseif sense == 3
        return scalar_set(Val(3), rhs)
    elseif sense == 4
        return scalar_set(Val(4), rhs)
    end
    error("Invalid constraint sense: ", sense)
end

function vector_set(sense::Integer, dimension::Integer)
    if sense == 0
        return MOI.Nonpositives(dimension)
    elseif sense == 1
        return MOI.Nonnegatives(dimension)
    elseif sense == 2
        return MOI.Zeros(dimension)
    end
    error("Invalid constraint group sense: ", sense)
end

# Never turn an expression into a VariableIndex bound: even x - 0 >= 0
# must remain an affine row, so it can coexist with a variable lower bound.
# Functions already passed as VariableIndex are bounds and stay unchanged.
function simplify(func::MOI.ScalarNonlinearFunction)
    f = MOI.Nonlinear.SymbolicAD.simplify(func)
    if f isa MOI.VariableIndex || f isa Float64
        return convert(MOI.ScalarAffineFunction{Float64}, f)
    end
    return f
end
simplify(func) = func

function normalize_and_add_constraint(model, func, sense::Integer, rhs::Float64)
    # Keep a concrete set type at each call site for experimental trimming.
    # Returning a runtime-selected set before this call loses that property.
    if sense == 0
        return MOI.Utilities.normalize_and_add_constraint(model, func, scalar_set(Val(0), rhs))
    elseif sense == 1
        return MOI.Utilities.normalize_and_add_constraint(model, func, scalar_set(Val(1), rhs))
    elseif sense == 2
        return MOI.Utilities.normalize_and_add_constraint(model, func, scalar_set(Val(2), rhs))
    elseif sense == 3
        return MOI.Utilities.normalize_and_add_constraint(model, func, scalar_set(Val(3), rhs))
    elseif sense == 4
        return MOI.Utilities.normalize_and_add_constraint(model, func, scalar_set(Val(4), rhs))
    end
    error("Invalid constraint sense: ", sense)
end

end # module JuMPyMOI
