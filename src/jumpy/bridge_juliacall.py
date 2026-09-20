"""
The JuliaCall implementation of native JuMP operations.

Each method calls Julia through juliacall. The compiled (juliac)
backend implements the same ops against the C entry points of the shared
library (jumpy.backend.JuliacOps).
"""

from __future__ import annotations

from pathlib import Path

from jumpy import backend

_JL = None


def _julia():
    global _JL
    if backend._LIB is not None:
        raise RuntimeError(
            "Compiled HiGHS is already initialized. Use one JuMPy backend per process; "
            "restart Python to switch to jumpy.juliacall."
        )
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
    for pkg in ["MathOptInterface", "HiGHS", "GenOpt", "JuMP"]:
        jl.seval(f"""
            if !haskey(Pkg.project().dependencies, "{pkg}")
                Pkg.add("{pkg}")
            end
        """)
    jl.seval("import MathOptInterface as MOI")
    jl.seval("import GenOpt")
    jl.seval("import HiGHS")
    jl.seval("import JuMP")
    # Load the same constructor source that is compiled into the JuliaC image.
    source = Path(__file__).with_name("julia") / "JuMPyModel.jl"
    if not source.is_file():
        # Editable installation: the canonical source stays in the Julia project.
        source = Path(__file__).resolve().parents[2] / "julia" / "src" / "JuMPyModel.jl"
    jl.include(str(source))
    _JL = jl
    return jl


class JuliaCallOps:
    def __init__(self):
        jl = _julia()
        self._jl = jl
        self.MOI = jl.MOI
        self._jump = jl.JuMPyModel
        # jl.Any[...] is broken in PythonCall with Julia 1.12+
        self._any_vec = jl.seval("(args...) -> Any[args...]")
        self._model = self._jump.model(jl.HiGHS.Optimizer())

    def free(self):
        pass  # the JuMP model is garbage-collected with this object

    # -- Native Julia objects -------------------------------------------------

    def constant(self, value):
        return value

    def apply(self, op, args):
        return self._jump.apply(self._jl.Symbol(op), self._any_vec(*args))

    def iterator(self, values):
        return self._jump.iterator(self._any_vec(*values))

    def float_array(self, values):
        return self._jl.seval("Vector{Float64}")(values)

    # -- Model building ----------------------------------------------------------

    def add_variables(self, count):
        return self._jump.add_variables(self._model, count)

    def add_constraint(self, func, set_):
        if not self._jl.isa(set_, self._jl.MOI.AbstractScalarSet):
            raise TypeError("Expected a native MOI scalar set from jumpy.juliacall")
        self._jump.add_constraint(self._model, func, set_)

    def set_objective(self, sense, func):
        self._jump.set_objective_sense(
            self._model, self.MOI.MIN_SENSE if sense == "min" else self.MOI.MAX_SENSE
        )
        self._jump.set_objective_function(self._model, func)

    def optimize(self):
        self._jump.optimize(self._model)
        return int(self._jl.Int(self._jl.JuMP.termination_status(self._model)))

    def value(self, func):
        return float(self._jump.value(func))
