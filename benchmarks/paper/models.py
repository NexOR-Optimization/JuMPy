"""Iterator-based ports of jump-dev/JuMPPaperBenchmarks at b4a3479.

The Julia reference formulations are the authority; the paper's Pyomo LQCP
objective has different parentheses. Array indices below match Julia's
column-major variable creation order so coefficient-level validation is possible.
"""

import math

from jumpy import minimize


def facility(model, n):
    if n < 1:
        raise ValueError("facility size must be positive")
    side = n + 1
    cells = side * side
    count = cells * n
    y = model.variables(2 * n, lower=0, upper=1)
    s = model.variables(count, lower=0)
    z = model.variables(count, binary=True)
    r = model.variables(2 * count)
    d = model.variable()
    model.objective = minimize(d)

    cell = model.iterator(range(cells))
    # Sum over facilities while retaining the customer iterator. This loop has
    # n terms, not n^3 constraints; GenOpt expands the resulting row template.
    assignment = z[cell]
    for facility_index in range(1, n):
        assignment = assignment + z[cell + cells * facility_index]
    model.constraint_group(assignment == 1)

    f = model.iterator(range(n))
    position = cell + cells * f
    x_coord = model.parameter([(q % side) / n for q in range(cells)])
    y_coord = model.parameter([(q // side) / n for q in range(cells)])
    big_m = 2 * math.sqrt(2)
    model.constraint_group(s[position] == d + big_m * (1 - z[position]))
    model.constraint_group(r[position] == x_coord[cell] - y[f])
    model.constraint_group(r[position + count] == y_coord[cell] - y[f + n])
    model.constraint_group(
        r[position] ** 2 + r[position + count] ** 2 <= s[position] ** 2
    )
    return model


def lqcp(model, n):
    if n < 3:
        raise ValueError("lqcp size must be at least three")
    side = n + 1
    dx = 1.0 / n
    dt = 1.58 / n
    h2 = dx * dx
    alpha = 0.001
    y = model.variables(side * side, lower=0, upper=1)
    u = model.variables(n, lower=-1, upper=1)
    target = model.parameter([0.5 * (1 - (j * dx) ** 2) for j in range(side)])
    j = model.iterator(range(1, n))
    i = model.iterator(range(1, n))
    tracking = (
        (y[n] - target[0]) ** 2
        + 2 * model.lazy_sum((y[n + side * j] - target[j]) ** 2)
        + (y[n + side * n] - target[n]) ** 2
    )
    control = 2 * model.lazy_sum(u[i - 1] ** 2) + u[n - 1] ** 2
    model.objective = minimize(0.25 * dx * tracking + 0.25 * alpha * dt * control)

    time = model.iterator(range(n))
    space = model.iterator(range(1, n))
    current = time + side * space
    model.constraint_group(
        h2 * (y[current + 1] - y[current])
        == 0.5 * dt * (
            y[current - side] - 2 * y[current] + y[current + side]
            + y[current + 1 - side] - 2 * y[current + 1]
            + y[current + 1 + side]
        )
    )
    initial = model.iterator(range(side))
    model.constraint_group(y[side * initial] == 0)
    boundary = model.iterator(range(1, side))
    model.constraint_group(
        y[boundary + 2 * side] - 4 * y[boundary + side] + 3 * y[boundary] == 0
    )
    model.constraint_group(
        y[boundary + (n - 2) * side]
        - 4 * y[boundary + (n - 1) * side]
        + 3 * y[boundary + n * side]
        == 2 * dx * (u[boundary - 1] - y[boundary + n * side])
    )
    return model


def latency(model, n=None):
    x = model.variable(lower=0)
    y = model.variable(lower=0, upper=3)
    model.objective = minimize(12 * x + 20 * y)
    model.constraint(6 * x + 8 * y >= 100)
    model.constraint(7 * x + 12 * y >= 120)
    return model


BUILDERS = {"facility": facility, "lqcp": lqcp, "latency": latency}
PAPER_SIZES = {"facility": (25, 50, 75, 100), "lqcp": (500, 1000, 1500, 2000)}
