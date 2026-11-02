# Copyright (c) 2022: Miles Lubin and contributors
# Copyright (c) 2026: NexOR Optimization SRL
# MIT license; see LICENSE in this directory.

module PaperModels

using JuMP

function facility(model, n)
    G = F = n
    @variables(model, begin
        0 <= y[1:F, 1:2] <= 1
        s[0:G, 0:G, 1:F] >= 0
        z[0:G, 0:G, 1:F], Bin
        r[0:G, 0:G, 1:F, 1:2]
        d
    end)
    @objective(model, Min, d)
    @constraint(model, [i in 0:G, j in 0:G], sum(z[i, j, f] for f = 1:F) == 1)
    M = 2 * sqrt(2)
    for i = 0:G, j = 0:G, f = 1:F
        @constraints(model, begin
            s[i, j, f] == d + M * (1 - z[i, j, f])
            r[i, j, f, 1] == i / G - y[f, 1]
            r[i, j, f, 2] == j / G - y[f, 2]
            sum(r[i, j, f, k]^2 for k = 1:2) <= s[i, j, f]^2
        end)
    end
    return model
end

function lqcp(model, n)
    m = n
    n1 = n - 1
    m1 = m - 1
    dx = 1 / n
    dt = 1.58 / m
    h2 = dx^2
    a = 0.001
    yt = [0.5 * (1 - (j * dx)^2) for j = 0:n]
    @variables(model, begin
        0 <= y[0:m, 0:n] <= 1
        -1 <= u[1:m] <= 1
    end)
    @objective(
        model,
        Min,
        1 / 4 *
        dx *
        (
            (y[m, 0] - yt[1])^2 +
            2 * sum((y[m, j] - yt[j+1])^2 for j = 1:n1) +
            (y[m, n] - yt[n+1])^2
        ) + 1 / 4 * a * dt * (2 * sum(u[i]^2 for i = 1:m1) + u[m]^2)
    )
    for i = 0:m1, j = 1:n1
        @constraint(
            model,
            h2 * (y[i+1, j] - y[i, j]) ==
            0.5 *
            dt *
            (
                y[i, j-1] - 2 * y[i, j] + y[i, j+1] + y[i+1, j-1] - 2 * y[i+1, j] +
                y[i+1, j+1]
            )
        )
    end
    for j = 0:n
        @constraint(model, y[0, j] == 0)
    end
    for i = 1:m
        @constraint(model, y[i, 2] - 4 * y[i, 1] + 3 * y[i, 0] == 0)
        @constraint(
            model,
            y[i, n-2] - 4 * y[i, n1] + 3 * y[i, n] == (2dx) * (u[i] - y[i, n])
        )
    end
    return model
end

function latency(model, n = nothing)
    @variable(model, x >= 0)
    @variable(model, 0 <= y <= 3)
    @objective(model, Min, 12x + 20y)
    @constraint(model, 6x + 8y >= 100)
    @constraint(model, 7x + 12y >= 120)
    return model
end

end
