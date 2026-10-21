# Minimal solver path: no JuMP, GenOpt, or bridge optimizer.
import HiGHS, MathOptInterface as MOI

Base.@ccallable function raw_highs_lp(rhs::Cdouble)::Cdouble
    model = HiGHS.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(rhs))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    return MOI.get(model, MOI.VariablePrimal(), x)
end
