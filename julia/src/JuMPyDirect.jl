module JuMPyDirect

# The compiled backend deliberately has no cache or bridge optimizer. JuMP owns
# its expressions; the selected MOI backend owns storage and solving.
import GenOpt
import JuMP
import MathOptInterface as MOI

function model(optimizer)
    result = JuMP.concrete_direct_model(optimizer)
    MOI.set(optimizer, MOI.Silent(), true)
    return result
end

add_variables(model, count) = JuMP.@variable(model, [1:count])

_solver_function(func) = func
_solver_function(func::GenOpt.PolynomialFunction{F}) where {F} = convert(F, func)

function add_constraint(model, func, set)
    constraint = JuMP.build_constraint(error, JuMP._MA.copy_if_mutable(func), set)
    constraint = JuMP.model_convert(model, constraint)
    JuMP.check_belongs_to_model(constraint, model)
    return _add_constraint(model, constraint)
end

function _add_constraint(model, constraint::JuMP.ScalarConstraint{F,S}) where {F,S}
    backend = JuMP.backend(model)
    func = _solver_function(JuMP.moi_function(model, constraint))
    M = typeof(func)
    MOI.supports_constraint(backend, M, S) || throw(MOI.UnsupportedConstraint{M,S}())
    index = MOI.Utilities.normalize_and_add_constraint(
        backend,
        func,
        JuMP.moi_set(constraint);
        allow_modify_function = true,
    )
    model.is_model_dirty = true
    return index
end

function _add_constraint(model, constraint::GenOpt.IteratedConstraint)
    T = JuMP.value_type(typeof(model))
    S = MOI.Utilities.scalar_set_type(typeof(JuMP.moi_set(constraint)), T)
    indices = GenOpt.add_constraints(
        JuMP.backend(model),
        JuMP.moi_function(model, constraint),
        S(zero(T)),
    )
    model.is_model_dirty = true
    return indices
end

function set_objective_sense(model, sense)
    MOI.set(JuMP.backend(model), MOI.ObjectiveSense(), sense)
    model.is_model_dirty = true
    return
end

# HiGHS accepts affine/quadratic objectives, not a bare MOI.VariableIndex.
_objective_function(model, func::JuMP.AbstractVariableRef) =
    JuMP.moi_function(model, 1.0 * func)
_objective_function(model, func::Real) =
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], Float64(func))
_objective_function(model, func::JuMP.AbstractJuMPScalar) = JuMP.moi_function(model, func)

function set_objective_function(model, func)
    func isa Real || JuMP.check_belongs_to_model(func, model)
    moi_func = _solver_function(_objective_function(model, func))
    attr = MOI.ObjectiveFunction{typeof(moi_func)}()
    backend = JuMP.backend(model)
    MOI.supports(backend, attr) || throw(MOI.UnsupportedAttribute(attr))
    MOI.set(backend, attr, moi_func)
    model.is_model_dirty = true
    return
end

function optimize(model)
    MOI.optimize!(JuMP.backend(model))
    model.is_model_dirty = false
    return
end

function _variable_value(variable::JuMP.AbstractVariableRef)
    model = JuMP.owner_model(variable)
    model.is_model_dirty && throw(JuMP.OptimizeNotCalled())
    backend = JuMP.backend(model)
    MOI.get(backend, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED &&
        throw(JuMP.OptimizeNotCalled())
    return MOI.get(backend, MOI.VariablePrimal(), JuMP.index(variable))
end

value(func::Real) = Float64(func)
value(func::JuMP.AbstractJuMPScalar) = JuMP.value(_variable_value, func)

end # module JuMPyDirect
