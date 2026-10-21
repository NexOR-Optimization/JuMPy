module StaticBridgeProbe
import GenOpt, HiGHS, MathOptInterface as MOI

function optimizer()
    inner = MOI.Bridges.Objective.Functionize{Float64}(HiGHS.Optimizer())
    model = MOI.Bridges.Constraint.SingleBridgeOptimizer{
        GenOpt.FunctionGeneratorBridge{Float64},
    }(inner)
    MOI.set(model, MOI.Silent(), true)
    return model
end

function solve(model, x)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    return MOI.get(model, MOI.VariablePrimal(), x)
end
end
