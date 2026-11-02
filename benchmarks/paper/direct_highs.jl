# A deliberately hardcoded LQCP lower bound: flat arrays and HiGHS.jl's C API.
module DirectHiGHSBenchmark

import HiGHS
import JSON

function lqcp_data(n::Int)
    n >= 3 || throw(ArgumentError("LQCP size must be at least three"))
    side = n + 1
    ny, nv, nr = side^2, side^2 + n, side^2
    dx, dt = 1.0 / n, 1.58 / n
    h2, off = dx^2, -0.5 * dt
    starts = Vector{HiGHS.HighsInt}(undef, nr + 1)
    columns = Vector{HiGHS.HighsInt}(undef, 6n * (n - 1) + side + 7n)
    values = Vector{Float64}(undef, length(columns))
    row, position = 1, 1
    @inbounds for j = 1:(n-1), t = 0:(n-1)
        c = t + side * j
        starts[row] = position - 1
        columns[position] = c - side
        columns[position+1] = c + 1 - side
        columns[position+2] = c
        columns[position+3] = c + 1
        columns[position+4] = c + side
        columns[position+5] = c + 1 + side
        values[position] = off
        values[position+1] = off
        values[position+2] = dt - h2
        values[position+3] = dt + h2
        values[position+4] = off
        values[position+5] = off
        position += 6
        row += 1
    end
    @inbounds for j = 0:n
        starts[row] = position - 1
        columns[position] = side * j
        values[position] = 1.0
        position += 1
        row += 1
    end
    @inbounds for t = 1:n
        starts[row] = position - 1
        columns[position] = t
        columns[position+1] = t + side
        columns[position+2] = t + 2side
        values[position] = 3.0
        values[position+1] = -4.0
        values[position+2] = 1.0
        position += 3
        row += 1
    end
    @inbounds for t = 1:n
        starts[row] = position - 1
        columns[position] = t + (n - 2) * side
        columns[position+1] = t + (n - 1) * side
        columns[position+2] = t + n * side
        columns[position+3] = ny + t - 1
        values[position] = 1.0
        values[position+1] = -4.0
        values[position+2] = 3.0 + 2dx
        values[position+3] = -2dx
        position += 4
        row += 1
    end
    starts[end] = position - 1
    @assert row == nr + 1 && position == length(values) + 1

    costs, lower, upper = zeros(nv), zeros(nv), ones(nv)
    lower[(ny+1):end] .= -1.0
    qstarts = Vector{HiGHS.HighsInt}(undef, nv + 1)
    qindices = Vector{HiGHS.HighsInt}(undef, 2n + 1)
    qvalues = Vector{Float64}(undef, 2n + 1)
    qposition, offset = 1, 0.0
    @inbounds for j = 0:n
        for t = 0:n
            qstarts[side*j+t+1] = qposition - 1
        end
        col = side * j + n
        target = 0.5 * (1 - (j * dx)^2)
        weight = (j == 0 || j == n) ? 0.25dx : 0.5dx
        costs[col+1] = -2 * weight * target
        offset += weight * target^2
        qindices[qposition] = col
        qvalues[qposition] = 2weight
        qposition += 1
    end
    @inbounds for i = 1:n
        qstarts[ny+i] = qposition - 1
        qindices[qposition] = ny + i - 1
        qvalues[qposition] = (i == n ? 0.5 : 1.0) * 0.001 * dt
        qposition += 1
    end
    qstarts[end] = qposition - 1
    return (;
        nv,
        nr,
        starts,
        columns,
        values,
        costs,
        lower,
        upper,
        rhs = zeros(nr),
        qstarts,
        qindices,
        qvalues,
        offset,
    )
end

function check(status)
    status == HiGHS.kHighsStatusError && error("HiGHS C API call failed")
    return
end

function construct(n::Int; solve::Bool = false)
    started = time_ns()
    data = lqcp_data(n)
    assembled = time_ns()
    highs = HiGHS.Highs_create()
    highs == C_NULL && error("Could not create HiGHS")
    try
        check(HiGHS.Highs_setBoolOptionValue(highs, "output_flag", false))
        check(HiGHS.Highs_setIntOptionValue(highs, "threads", 1))
        check(HiGHS.Highs_setStringOptionValue(highs, "presolve", "off"))
        if !solve
            check(HiGHS.Highs_setDoubleOptionValue(highs, "time_limit", 0.0))
        end
        check(
            HiGHS.Highs_passModel(
                highs,
                data.nv,
                data.nr,
                length(data.values),
                length(data.qvalues),
                HiGHS.kHighsMatrixFormatRowwise,
                HiGHS.kHighsHessianFormatTriangular,
                HiGHS.kHighsObjSenseMinimize,
                data.offset,
                data.costs,
                data.lower,
                data.upper,
                data.rhs,
                data.rhs,
                data.starts,
                data.columns,
                data.values,
                data.qstarts,
                data.qindices,
                data.qvalues,
                C_NULL,
            ),
        )
        loaded = time_ns()
        check(HiGHS.Highs_run(highs))
        finished = time_ns()
        return highs,
        (;
            seconds = (finished - started) / 1e9,
            assembly_seconds = (assembled - started) / 1e9,
            load_seconds = (loaded - assembled) / 1e9,
            run_seconds = (finished - loaded) / 1e9,
        )
    catch
        HiGHS.Highs_destroy(highs)
        rethrow()
    end
end

function main(sizes = [100, 250, 500, 1000]; repeats = 5)
    highs, _ = construct(5)
    HiGHS.Highs_destroy(highs)
    for n in sizes, sample = 1:repeats
        highs, timings = construct(n)
        try
            println(
                JSON.json((;
                    event = "sample",
                    frontend = "hardcoded-highs-julia",
                    backend = "highs",
                    model = "lqcp",
                    size = n,
                    sample,
                    variables = HiGHS.Highs_getNumCol(highs),
                    timings...,
                    julia = string(VERSION),
                    highs_jl = string(pkgversion(HiGHS)),
                    highs_version = unsafe_string(HiGHS.Highs_version()),
                )),
            )
            flush(stdout)
        finally
            HiGHS.Highs_destroy(highs)
        end
    end
    return
end

end # module DirectHiGHSBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    sizes = isempty(ARGS) ? [100, 250, 500, 1000] : parse.(Int, ARGS)
    DirectHiGHSBenchmark.main(sizes)
end
