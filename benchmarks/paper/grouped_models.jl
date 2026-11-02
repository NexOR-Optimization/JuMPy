# The same flattened formulations as models.py, with Julia building the
# templates. Reuse the compiled backend's direct adapter and bulk bounds so
# this comparison measures the cost of sending templates from Python.
module GroupedPaperModels

import GenOpt
import JuMP
import MathOptInterface as MOI
include("../../julia/src/JuMPyDirect.jl")
include("../../julia/src/JuMPyModel.jl")

function variables(model, count; lower = nothing, upper = nothing, binary = false)
    result = JuMPyDirect.add_variables(model, count)
    for (value, constructor) in ((lower, MOI.GreaterThan), (upper, MOI.LessThan))
        if value !== nothing
            JuMPyModel.add_variable_constraints(model, result, constructor(Float64(value)))
        end
    end
    if binary
        JuMPyModel.add_variable_constraints(model, result, MOI.ZeroOne())
    end
    return result
end

function objective(model, func)
    JuMPyDirect.set_objective_sense(model, MOI.MIN_SENSE)
    JuMPyDirect.set_objective_function(model, func)
    return
end

equal(model, func) = JuMPyDirect.add_constraint(model, func, MOI.EqualTo(0.0))

function lqcp(model, n)
    n >= 3 || throw(ArgumentError("lqcp size must be at least three"))
    side = n + 1
    dx, dt, alpha = 1.0 / n, 1.58 / n, 0.001
    h2 = dx * dx
    y = variables(model, side * side; lower = 0, upper = 1)
    u = variables(model, n; lower = -1, upper = 1)
    target = [0.5 * (1 - (j * dx)^2) for j = 0:n]
    j = GenOpt.iterator(1:(n-1))
    i = GenOpt.iterator(1:(n-1))
    tracking =
        (y[n+1] - target[1])^2 +
        2 * GenOpt.LazySum((y[n+side*j+1] - target[j+1])^2) +
        (y[n+side*n+1] - target[n+1])^2
    control = 2 * GenOpt.LazySum(u[i]^2) + u[n]^2
    objective(model, 0.25 * dx * tracking + 0.25 * alpha * dt * control)

    time = GenOpt.iterator(0:(n-1))
    space = GenOpt.iterator(1:(n-1))
    current = time + side * space + 1
    equal(
        model,
        h2 * (y[current+1] - y[current]) -
        0.5 *
        dt *
        (
            y[current-side] - 2 * y[current] + y[current+side] + y[current+1-side] -
            2 * y[current+1] + y[current+1+side]
        ),
    )
    initial = GenOpt.iterator(0:n)
    equal(model, y[side*initial+1])
    boundary = GenOpt.iterator(1:n)
    equal(model, y[boundary+2*side+1] - 4 * y[boundary+side+1] + 3 * y[boundary+1])
    equal(
        model,
        y[boundary+(n-2)*side+1] - 4 * y[boundary+(n-1)*side+1] + 3 * y[boundary+n*side+1] -
        2 * dx * (u[boundary] - y[boundary+n*side+1]),
    )
    return model
end

function facility(model, n)
    n >= 1 || throw(ArgumentError("facility size must be positive"))
    side, cells = n + 1, (n + 1)^2
    count = cells * n
    y = variables(model, 2n; lower = 0, upper = 1)
    s = variables(model, count; lower = 0)
    z = variables(model, count; binary = true)
    r = variables(model, 2count)
    d = only(variables(model, 1))
    objective(model, d)
    cell = GenOpt.iterator(0:(cells-1))
    assignment = z[cell+1]
    for f = 1:(n-1)
        assignment = assignment + z[cell+cells*f+1]
    end
    equal(model, assignment - 1)
    f = GenOpt.iterator(0:(n-1))
    position = cell + cells * f + 1
    x_coord = [(q % side) / n for q = 0:(cells-1)]
    y_coord = [(q ÷ side) / n for q = 0:(cells-1)]
    big_m = 2 * sqrt(2)
    equal(model, s[position] - (d + big_m * (1 - z[position])))
    equal(model, r[position] - (x_coord[cell+1] - y[f+1]))
    equal(model, r[position+count] - (y_coord[cell+1] - y[f+n+1]))
    JuMPyDirect.add_constraint(
        model,
        r[position]^2 + r[position+count]^2 - s[position]^2,
        MOI.LessThan(0.0),
    )
    return model
end

end
