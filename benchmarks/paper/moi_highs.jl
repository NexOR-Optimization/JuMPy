# Standalone LQCP: public MOI operations on HiGHS.Optimizer only.
module MOIHiGHSBenchmark

import HiGHS
import JSON
import MathOptInterface as MOI

const F = MOI.ScalarAffineFunction{Float64}
const Term = MOI.ScalarAffineTerm{Float64}

function add_constraints(model, functions, sets, bulk)
    if bulk
        return MOI.add_constraints(model, functions, sets)
    end
    return [MOI.add_constraint(model, f, s) for (f, s) in zip(functions, sets)]
end

function add_rows(model, builder, count, bulk, batch_size)
    for first = 1:batch_size:count
        last = min(first + batch_size - 1, count)
        functions = [builder(row) for row = first:last]
        add_constraints(model, functions, fill(MOI.EqualTo(0.0), length(functions)), bulk)
    end
    return
end

function lqcp(model, n; bulk = true, batch_size = 65_536)
    n >= 3 || throw(ArgumentError("LQCP size must be at least three"))
    batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
    started = time_ns()
    side, ny = n + 1, (n + 1)^2
    nv = ny + n
    dx, dt, alpha = 1.0 / n, 1.58 / n, 0.001
    h2, off = dx^2, -0.5dt
    variables = bulk ? MOI.add_variables(model, nv) : [MOI.add_variable(model) for _ = 1:nv]
    variables_done = time_ns()
    lower = [MOI.GreaterThan(i <= ny ? 0.0 : -1.0) for i = 1:nv]
    add_constraints(model, variables, lower, bulk)
    add_constraints(model, variables, fill(MOI.LessThan(1.0), nv), bulk)
    bounds_done = time_ns()

    quadratic = MOI.ScalarQuadraticTerm{Float64}[]
    affine = Term[]
    offset = 0.0
    for j = 0:n
        variable = variables[side*j+n+1]
        target = 0.5 * (1 - (j * dx)^2)
        weight = (j == 0 || j == n) ? 0.25dx : 0.5dx
        push!(quadratic, MOI.ScalarQuadraticTerm(2weight, variable, variable))
        push!(affine, Term(-2weight * target, variable))
        offset += weight * target^2
    end
    for i = 1:n
        variable = variables[ny+i]
        coefficient = (i == n ? 0.5 : 1.0) * alpha * dt
        push!(quadratic, MOI.ScalarQuadraticTerm(coefficient, variable, variable))
    end
    objective = MOI.ScalarQuadraticFunction(quadratic, affine, offset)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    objective_done = time_ns()

    # Canonical six-term PDE stencil; time changes fastest within each column.
    pde = function (row)
        j, t = divrem(row - 1, n)
        c = t + side * (j + 1) + 1
        return F(
            Term[
                Term(off, variables[c-side]),
                Term(off, variables[c+1-side]),
                Term(dt - h2, variables[c]),
                Term(dt + h2, variables[c+1]),
                Term(off, variables[c+side]),
                Term(off, variables[c+1+side]),
            ],
            0.0,
        )
    end
    add_rows(model, pde, n * (n - 1), bulk, batch_size)
    initial = row -> F(Term[Term(1.0, variables[side*(row-1)+1])], 0.0)
    add_rows(model, initial, side, bulk, batch_size)
    left =
        t -> F(
            Term[
                Term(3.0, variables[t+1]),
                Term(-4.0, variables[t+side+1]),
                Term(1.0, variables[t+2side+1]),
            ],
            0.0,
        )
    add_rows(model, left, n, bulk, batch_size)
    right =
        t -> F(
            Term[
                Term(1.0, variables[t+(n-2)*side+1]),
                Term(-4.0, variables[t+(n-1)*side+1]),
                Term(3.0 + 2dx, variables[t+n*side+1]),
                Term(-2dx, variables[ny+t]),
            ],
            0.0,
        )
    add_rows(model, right, n, bulk, batch_size)
    finished = time_ns()
    return (;
        variables_seconds = (variables_done - started) / 1e9,
        bounds_seconds = (bounds_done - variables_done) / 1e9,
        objective_seconds = (objective_done - bounds_done) / 1e9,
        rows_seconds = (finished - objective_done) / 1e9,
    )
end

function construct(n; bulk = true, solve = false, batch_size = 65_536)
    started = time_ns()
    model = HiGHS.Optimizer()
    try
        MOI.set(model, MOI.Silent(), true)
        MOI.set(model, MOI.NumberOfThreads(), 1)
        MOI.set(model, MOI.RawOptimizerAttribute("presolve"), "off")
        solve || MOI.set(model, MOI.TimeLimitSec(), 0.0)
        setup_done = time_ns()
        phases = lqcp(model, n; bulk, batch_size)
        built = time_ns()
        MOI.optimize!(model)
        finished = time_ns()
        return model,
        (;
            seconds = (finished - started) / 1e9,
            setup_seconds = (setup_done - started) / 1e9,
            construction_seconds = (built - setup_done) / 1e9,
            phases...,
            run_seconds = (finished - built) / 1e9,
        )
    catch
        finalize(model)
        rethrow()
    end
end

function bulk_methods()
    return (;
        variables = which(MOI.add_variables, (HiGHS.Optimizer, Int)).module === HiGHS,
        bounds = which(
            MOI.add_constraints,
            (HiGHS.Optimizer, Vector{MOI.VariableIndex}, Vector{MOI.GreaterThan{Float64}}),
        ).module === HiGHS,
        rows = which(
            MOI.add_constraints,
            (HiGHS.Optimizer, Vector{F}, Vector{MOI.EqualTo{Float64}}),
        ).module === HiGHS,
    )
end

function main(sizes = [1000]; repeats = 5, modes = (true, false), batch_size = 65_536)
    println(
        JSON.json((;
            event = "environment",
            julia = string(VERSION),
            highs_jl = string(pkgversion(HiGHS)),
            moi = string(pkgversion(MOI)),
            highs_source = pathof(HiGHS),
            native_bulk_methods = bulk_methods(),
        )),
    )
    for bulk in modes
        warm, _ = construct(first(sizes); bulk, batch_size)
        finalize(warm)
        warm = nothing
        for n in sizes, sample = 1:repeats
            GC.gc(true)
            model, timings = construct(n; bulk, batch_size)
            try
                println(
                    JSON.json((;
                        event = "sample",
                        mode = bulk ? "bulk" : "scalar",
                        model = "lqcp",
                        size = n,
                        sample,
                        batch_size,
                        variables = MOI.get(model, MOI.NumberOfVariables()),
                        constraints = MOI.get(
                            model,
                            MOI.NumberOfConstraints{F,MOI.EqualTo{Float64}}(),
                        ),
                        timings...,
                    )),
                )
                flush(stdout)
            finally
                finalize(model)
            end
            model = nothing
        end
    end
    return
end

end # module MOIHiGHSBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    MOIHiGHSBenchmark.main(isempty(ARGS) ? [1000] : parse.(Int, ARGS))
end
