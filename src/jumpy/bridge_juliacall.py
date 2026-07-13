"""
The juliacall implementation of the MOI ops.

Each method is one MOI call through juliacall. The compiled (juliac)
backend implements the same ops against the C entry points of the shared
library (jumpy.backend.JuliacOps).
"""

from __future__ import annotations

_JL = None


def _julia():
    global _JL
    if _JL is not None:
        return _JL
    try:
        from juliacall import Main as jl
    except ImportError:
        raise ImportError(
            "juliacall is not installed.\n"
            "Install it with: pip install jumpy[juliacall]\n"
            "This will also install Julia automatically if needed."
        ) from None
    # Install and load Julia packages on first use
    jl.seval("using Pkg")
    for pkg in ["MathOptInterface", "HiGHS", "GenOpt"]:
        jl.seval(f"""
            if !haskey(Pkg.project().dependencies, "{pkg}")
                Pkg.add("{pkg}")
            end
        """)
    jl.seval("import MathOptInterface as MOI")
    jl.seval("import GenOpt")
    jl.seval("import HiGHS")
    _JL = jl
    return jl


class JuliaCallOps:
    def __init__(self):
        jl = _julia()
        self._jl = jl
        # jl.Any[...] is broken in PythonCall with Julia 1.12+
        self._any_vec = jl.seval("(args...) -> Any[args...]")
        # {} type application is not expressible in Python syntax
        self._objective_attr = jl.seval("f -> MOI.ObjectiveFunction{typeof(f)}()")
        self._to_affine = jl.seval("f -> convert(MOI.ScalarAffineFunction{Float64}, f)")
        self._optimizer = jl.seval("""
            let optimizer = MOI.instantiate(
                    MOI.OptimizerWithAttributes(HiGHS.Optimizer, "output_flag" => false),
                    with_bridge_type = Float64,
                )
                MOI.Bridges.add_bridge(optimizer, GenOpt.FunctionGeneratorBridge{Float64})
                optimizer
            end
        """)
        self._variables = []

    def free(self):
        pass  # the optimizer is garbage-collected with this object

    # -- MOI functions ---------------------------------------------------------

    def constant(self, value):
        return value

    def variable(self, index):
        return self._variables[index]

    def scalar_nonlinear(self, head, args):
        jl = self._jl
        return jl.MOI.ScalarNonlinearFunction(jl.Symbol(head), self._any_vec(*args))

    def iterator(self, values):
        jl = self._jl
        return jl.GenOpt.IteratorRef(jl.GenOpt.Iterator(jl.seval("collect")(values)))

    def contiguous_variables(self, start, count):
        return self._jl.seval(
            f"GenOpt.ContiguousArrayOfVariables({start}, ({count},))"
        )

    def float_array(self, values):
        return self._jl.seval("collect")(values)

    def _simplify(self, func):
        """
        Narrow an affine ScalarNonlinearFunction to ScalarAffineFunction —
        but never below: `x >= 0` as a constraint must stay a row, not
        become a VariableIndex bound (same semantics as JuMP's @constraint).
        Only a function that already is a VariableIndex — the bounds path —
        is a bound.
        """
        jl = self._jl
        if jl.isa(func, jl.MOI.ScalarNonlinearFunction):
            func = jl.MOI.Nonlinear.SymbolicAD.simplify(func)
            if jl.isa(func, jl.MOI.VariableIndex) or isinstance(func, float):
                func = self._to_affine(func)
        return func

    # -- Model building ----------------------------------------------------------

    def add_variables(self, count):
        start = len(self._variables)
        self._variables.extend(self._jl.MOI.add_variables(self._optimizer, count))
        return start

    def _set(self, sense, rhs):
        jl = self._jl
        if sense == "<=":
            return jl.MOI.LessThan(float(rhs))
        if sense == ">=":
            return jl.MOI.GreaterThan(float(rhs))
        if sense == "==":
            return jl.MOI.EqualTo(float(rhs))
        if sense == "binary":
            return jl.MOI.ZeroOne()
        if sense == "integer":
            return jl.MOI.Integer()
        raise ValueError(f"Unknown constraint sense: {sense}")

    def add_constraint(self, func, sense, rhs):
        self._jl.MOI.Utilities.normalize_and_add_constraint(
            self._optimizer, self._simplify(func), self._set(sense, rhs),
        )

    def add_constraint_group(self, func, sense, linear):
        jl = self._jl
        if linear:
            target = "MOI.ScalarAffineFunction{Float64}"
        else:
            target = "MOI.ScalarNonlinearFunction"
        template, iterators = jl.GenOpt.collect_iterator_refs(func)
        generator = jl.seval(f"GenOpt.FunctionGenerator{{{target}}}")(template, iterators)
        n = jl.MOI.output_dimension(generator)
        if sense == "<=":
            set_ = jl.MOI.Nonpositives(n)
        elif sense == ">=":
            set_ = jl.MOI.Nonnegatives(n)
        elif sense == "==":
            set_ = jl.MOI.Zeros(n)
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")
        jl.MOI.add_constraint(self._optimizer, generator, set_)

    def set_objective(self, sense, func):
        jl = self._jl
        moi_sense = jl.MOI.MIN_SENSE if sense == "min" else jl.MOI.MAX_SENSE
        jl.MOI.set(self._optimizer, jl.MOI.ObjectiveSense(), moi_sense)
        func = self._simplify(func)
        jl.MOI.set(self._optimizer, self._objective_attr(func), func)

    def optimize(self):
        jl = self._jl
        jl.MOI.optimize_b(self._optimizer)
        return int(jl.Integer(jl.MOI.get(self._optimizer, jl.MOI.TerminationStatus())))

    def get_values(self, count):
        jl = self._jl
        return [
            float(jl.MOI.get(self._optimizer, jl.MOI.VariablePrimal(), v))
            for v in self._variables[:count]
        ]
