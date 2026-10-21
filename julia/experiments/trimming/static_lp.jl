include("static_common.jl")
using .StaticBridgeProbe: MOI, optimizer, solve

Base.@ccallable function static_lp(rhs::Cdouble)::Cdouble
    model = optimizer()
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(rhs))
    return solve(model, x)
end
