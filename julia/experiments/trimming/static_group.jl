include("static_common.jl")
using .StaticBridgeProbe: MOI, GenOpt, optimizer, solve

Base.@ccallable function static_group(rhs::Cdouble)::Cdouble
    model = optimizer()
    x = MOI.add_variable(model)
    template = MOI.ScalarNonlinearFunction(:-, Any[x, rhs])
    generator = GenOpt.FunctionGenerator{MOI.ScalarAffineFunction{Float64}}(
        template,
        GenOpt.Iterator[GenOpt.Iterator([1])],
    )
    MOI.add_constraint(model, generator, MOI.Nonnegatives(1))
    return solve(model, x)
end
