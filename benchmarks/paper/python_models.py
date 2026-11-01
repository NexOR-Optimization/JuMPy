"""LQCP ports using each library's ordinary in-memory HiGHS interface."""

from dataclasses import dataclass


@dataclass
class BuiltModel:
    model: object
    highs: object
    solver: object = None

    def close(self):
        self.highs.clear()
        self.model = self.highs = self.solver = None


def configure(highs, *, solve=False):
    import highspy

    options = dict(output_flag=False, threads=1, presolve="off")
    if not solve:
        options["time_limit"] = 0.0
    for key, value in options.items():
        if highs.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected {key}={value!r}")


def highspy_lqcp(n, *, solve=False):
    """Assemble sparse arrays in bulk; include their construction in the timer."""
    import highspy
    import numpy as np
    from scipy import sparse

    side = n + 1
    ny, nv = side * side, side * side + n
    dx, dt = 1.0 / n, 1.58 / n
    h2 = dx * dx
    time = np.repeat(np.arange(n), n - 1)
    space = np.tile(np.arange(1, n), n)
    current = time + side * space
    count = len(current)
    columns = np.column_stack((current-side, current, current+side,
                               current+1-side, current+1, current+1+side))
    coefficients = np.array([-0.5*dt, dt-h2, -0.5*dt, -0.5*dt, h2+dt, -0.5*dt])
    pde = sparse.coo_matrix((np.tile(coefficients, count),
                            (np.repeat(np.arange(count), 6), columns.ravel())), shape=(count, nv))
    initial = sparse.coo_matrix(
        (np.ones(side), (np.arange(side), side*np.arange(side))), shape=(side, nv))
    boundary = np.arange(1, side)
    left = sparse.coo_matrix((np.tile([1., -4., 3.], n),
                             (np.repeat(np.arange(n), 3),
                              np.column_stack((boundary+2*side, boundary+side, boundary)).ravel())), shape=(n, nv))
    right = sparse.coo_matrix((np.tile([1., -4., 3.+2*dx, -2*dx], n),
                              (np.repeat(np.arange(n), 4),
                               np.column_stack((boundary+(n-2)*side, boundary+(n-1)*side,
                                                boundary+n*side, ny+boundary-1)).ravel())), shape=(n, nv))
    matrix = sparse.vstack((pde, initial, left, right), format="csr")
    matrix.sum_duplicates()
    matrix.sort_indices()

    target = 0.5 * (1 - (np.arange(side)*dx)**2)
    tracking_weights = np.full(side, 0.5*dx)
    tracking_weights[[0, -1]] = 0.25*dx
    control_weights = np.full(n, 0.5*0.001*dt)
    control_weights[-1] = 0.25*0.001*dt
    terminal = n + side*np.arange(side)
    cost, diagonal = np.zeros(nv), np.zeros(nv)
    cost[terminal] = -2*tracking_weights*target
    diagonal[terminal] = 2*tracking_weights
    diagonal[ny:] = 2*control_weights

    data = highspy.HighsModel()
    lp = data.lp_
    lp.num_col_, lp.num_row_ = nv, side*side
    lp.col_cost_ = cost
    lp.col_lower_ = np.r_[np.zeros(ny), -np.ones(n)]
    lp.col_upper_ = np.ones(nv)
    lp.row_lower_ = lp.row_upper_ = np.zeros(side*side)
    lp.offset_ = float(np.dot(tracking_weights, target**2))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data
    hessian = data.hessian_
    hessian.dim_, hessian.format_ = nv, highspy.HessianFormat.kTriangular
    hessian.start_ = np.r_[0, np.cumsum(diagonal != 0)].astype(np.int32)
    hessian.index_ = np.flatnonzero(diagonal).astype(np.int32)
    hessian.value_ = diagonal[diagonal != 0]
    highs = highspy.Highs()
    configure(highs, solve=solve)
    if highs.passModel(data) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the bulk LQCP model")
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS failed to ingest the model")
    return BuiltModel(highs, highs)


def pyomo_lqcp(n, *, solve=False):
    import pyomo.environ as pyo
    from pyomo.contrib.solver.solvers.highs import Highs

    side, ny = n + 1, (n + 1)**2
    dx, dt = 1.0/n, 1.58/n
    h2 = dx*dx
    model = pyo.ConcreteModel()
    model.y = pyo.Var(range(ny), bounds=(0, 1))
    model.u = pyo.Var(range(n), bounds=(-1, 1))
    y, u = model.y, model.u
    tracking = pyo.quicksum((1 if j in (0, n) else 2) *
                             (y[n+side*j] - 0.5*(1-(j*dx)**2))**2 for j in range(side))
    control = pyo.quicksum((1 if i == n-1 else 2)*u[i]**2 for i in range(n))
    model.objective = pyo.Objective(expr=0.25*dx*tracking + 0.25*0.001*dt*control)

    def pde(m, t, j):
        c = t + side*j
        return h2*(y[c+1]-y[c]) == 0.5*dt*(
            y[c-side]-2*y[c]+y[c+side] + y[c+1-side]-2*y[c+1]+y[c+1+side])

    model.pde = pyo.Constraint(range(n), range(1, n), rule=pde)
    model.initial = pyo.Constraint(range(side), rule=lambda m, j: y[side*j] == 0)
    model.left = pyo.Constraint(range(1, side), rule=lambda m, t:
                                y[t+2*side]-4*y[t+side]+3*y[t] == 0)
    model.right = pyo.Constraint(range(1, side), rule=lambda m, t:
                                 y[t+(n-2)*side]-4*y[t+(n-1)*side]+3*y[t+n*side]
                                 == 2*dx*(u[t-1]-y[t+n*side]))
    solver = Highs()
    options = dict(output_flag=False, presolve="off")
    if not solve:
        options["time_limit"] = 0.0
    solver.solve(model, threads=1, load_solutions=False, symbolic_solver_labels=False,
                 raise_exception_on_nonoptimal_result=False, solver_options=options)
    # Public solve performs ingestion; inspect its native model for validation
    # and variable counts, outside the construction timer.
    return BuiltModel(model, solver._solver_model, solver)


def linopy_lqcp(n, *, solve=False):
    import highspy
    import linopy
    import numpy as np
    import pandas as pd
    import xarray as xr

    side = n + 1
    dx, dt = 1.0/n, 1.58/n
    space = pd.RangeIndex(side, name="space")
    time = pd.RangeIndex(side, name="time")
    model = linopy.Model(set_names_in_solver_io=False)
    # C order makes time the fastest dimension, matching JuMP's variable order.
    y = model.add_variables(0, 1, coords=[space, time], name="y")
    u = model.add_variables(-1, 1, coords=[pd.RangeIndex(1, side, name="time")], name="u")
    target = xr.DataArray(0.5*(1-(np.arange(side)*dx)**2),
                          dims="space", coords={"space": space})
    weights = xr.DataArray(np.r_[1., np.full(n-1, 2.), 1.],
                           dims="space", coords={"space": space})
    controls = xr.DataArray(np.r_[np.full(n-1, 2.), 1.],
                            dims="time", coords={"time": u.coords["time"]})
    terminal = y.sel(time=n, drop=True)
    # Linopy rejects objective constants. Preserve it in the native model below.
    model.add_objective(0.25*dx*((terminal*terminal - 2*terminal*target)*weights).sum() +
                        0.25*0.001*dt*((u*u)*controls).sum())
    laplacian = y.shift(space=1).fillna(0) - 2*y + y.shift(space=-1).fillna(0)
    pde = dx*dx*(y.shift(time=-1).fillna(0)-y) - 0.5*dt*(laplacian + laplacian.shift(time=-1).fillna(0))
    model.add_constraints(pde.isel(space=slice(1, n), time=slice(0, n)) == 0, name="pde")
    model.add_constraints(y.sel(time=0, drop=True) == 0, name="initial")

    def boundary(j):
        return y.sel(space=j, drop=True).isel(time=slice(1, side))

    model.add_constraints(boundary(2)-4*boundary(1)+3*boundary(0) == 0, name="left")
    model.add_constraints(boundary(n-2)-4*boundary(n-1)+3*boundary(n)
                          == 2*dx*(u-boundary(n)), name="right")
    highs = model.to_highspy(set_names=False)
    highs.changeObjectiveOffset(float(0.25*dx*(weights*target**2).sum()))
    configure(highs, solve=solve)
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS failed to ingest the Linopy model")
    return BuiltModel(model, highs)


BUILDERS = {"highspy": highspy_lqcp, "pyomo": pyomo_lqcp, "linopy": linopy_lqcp}


def construct(frontend, n, *, solve=False):
    if n < 3:
        raise ValueError("LQCP size must be at least three")
    return BUILDERS[frontend](n, solve=solve)
