module PaperStorage
import MathOptInterface as MOI

MOI.Utilities.@model(
    Model,
    (),
    (MOI.LessThan, MOI.GreaterThan, MOI.EqualTo),
    (),
    (),
    (),
    (MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction),
    (),
    (),
)
MOI.supports(::Model, ::MOI.Silent) = true
MOI.set(model::Model, ::MOI.Silent, value::Bool) = (model.ext[:silent] = value; nothing)
MOI.get(model::Model, ::MOI.Silent) = get(model.ext, :silent, false)
end
