"""
The juliacall implementation of the MOI ops.

Each method is one MOI call through juliacall. The compiled (juliac)
backend implements the same ops against the C entry points of the shared
library (jumpy.backend.JuliacOps).
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

from jumpy import backend
from jumpy.backend import _SENSE_CODES

_JL = None


def _configure_shared(lib):
    """Choose one image/runtime before JuliaCall imports, including C-first use."""
    backend._require_shared_image(lib)
    existing = sys.modules.get("juliacall")
    if existing is not None and existing.CONFIG.get("inited"):
        backend._check_shared_runtime(lib, existing.Main)
        return
    if lib.jl_is_initialized():
        backend._check_active_image(lib)

    exe = sys._xoptions.get("juliacall-exe", os.environ.get("PYTHON_JULIACALL_EXE"))
    project = sys._xoptions.get("juliacall-project", os.environ.get("PYTHON_JULIACALL_PROJECT"))
    if exe is None and project is None:
        import juliapkg

        exe, project = juliapkg.executable(), juliapkg.project()
    elif exe is None or project is None:
        raise RuntimeError("Set PYTHON_JULIACALL_EXE and PYTHON_JULIACALL_PROJECT together")
    installation = subprocess.run(
        [exe, "--startup-file=no", "-e", 'print(VERSION, "\\0", Sys.BINDIR, "\\0", Sys.STDLIB)'],
        check=True, capture_output=True, text=True,
    ).stdout.split("\0")
    version, bindir, stdlib = installation
    expected = os.fsdecode(lib.jl_ver_string())
    if version != expected:
        raise RuntimeError(
            f"The JuMPy image requires Julia {expected}, but JuliaCall selected {version}. "
            "Select a matching executable with PYTHON_JULIAPKG_EXE."
        )
    runtime = lib.jumpy_runtime_library_path()
    if runtime is None:
        raise RuntimeError("Could not locate the compiled library's Julia runtime")
    settings = {
        "EXE": str(exe),
        "PROJECT": str(project),
        "LIB": os.fsdecode(runtime),
        "SYSIMAGE": str(Path(lib._name).resolve()),
    }
    for key, value in settings.items():
        selected = sys._xoptions.get("juliacall-" + key.lower())
        if selected is None:
            selected = os.environ.get("PYTHON_JULIACALL_" + key)
        if selected is not None and Path(selected).resolve() != Path(value).resolve():
            raise RuntimeError(f"JuliaCall's {key.lower()} conflicts with the shared JuMPy image")
    for key, value in settings.items():
        os.environ["PYTHON_JULIACALL_" + key] = value

    if lib.jl_is_initialized():
        # JuliaCall's jl_init is a no-op now, so select its package environment
        # explicitly before it tries `using PythonCall`. No files are changed.
        project_file = Path(project)
        if not project_file.is_file():
            project_file /= "Project.toml"
        def literal(value):
            return json.dumps(str(value), ensure_ascii=False).replace("$", "\\$")

        # JuliaC bundles native libraries, not the full standard-library source
        # tree. A C-first initialization therefore needs the matching Julia
        # installation's paths before PythonCall can load additional packages.
        code = (
            f"Base.ACTIVE_PROJECT[] = {literal(project_file)}; "
            f"Sys.BINDIR = {literal(bindir)}; Sys.STDLIB = {literal(stdlib)}"
        )
        lib.jl_eval_string.argtypes = [backend.ctypes.c_char_p]
        lib.jl_eval_string.restype = backend.ctypes.c_void_p
        if not lib.jl_eval_string(code.encode()):
            raise RuntimeError("Could not select JuliaCall's project in the existing runtime")


def _julia(shared_lib=None):
    global _JL
    if backend._SHUTTING_DOWN:
        raise RuntimeError("The Julia runtime is shutting down")
    if shared_lib is None:
        shared_lib = backend._LIB
        path = backend.find_lib() if shared_lib is None else None
        if path is not None:
            shared_lib = backend._open_lib(path)
    if _JL is not None:
        if shared_lib is not None:
            backend._check_shared_runtime(shared_lib, _JL)
        return _JL
    if shared_lib is not None:
        backend._require_shared_image(shared_lib)
        # Give the same useful dependency error as the source-only backend.
        import importlib.util

        if importlib.util.find_spec("juliacall") is None:
            raise ImportError("Install JuliaCall with: pip install jumpy[juliacall]")
        _configure_shared(shared_lib)
    try:
        from juliacall import Main as jl
    except ImportError:
        raise ImportError(
            "juliacall is not installed.\n"
            "Install it with: pip install jumpy[juliacall]\n"
            "This will also install Julia automatically if needed."
        ) from None
    if shared_lib is not None:
        backend._check_shared_runtime(shared_lib, jl)
        jl.seval("""
            const MOI = JuMPyHiGHS.MOI
            const GenOpt = JuMPyHiGHS.GenOpt
            const HiGHS = JuMPyHiGHS.HiGHS
            const JuMPyMOI = JuMPyHiGHS.JuMPyMOI
        """)
        if backend._LIB is None:
            backend._init_lib(shared_lib._name)
        backend._arm_shutdown_guard()
        _JL = jl
        return jl
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
    # Load the same constructor source that is compiled into the JuliaC image.
    source = Path(__file__).with_name("julia") / "JuMPyMOI.jl"
    if not source.is_file():
        # Editable installation: the canonical source stays in the Julia project.
        source = Path(__file__).resolve().parents[2] / "julia" / "src" / "JuMPyMOI.jl"
    jl.include(str(source))
    _JL = jl
    backend._arm_shutdown_guard()
    return jl


class JuliaCallOps:
    def __init__(self):
        jl = _julia()
        self._jl = jl
        self._moi = jl.JuMPyMOI
        self._constructors = None
        if backend._LIB is not None:
            from jumpy.moi import MOIConstructors

            self._constructors = MOIConstructors(backend._LIB)
        # jl.Any[...] is broken in PythonCall with Julia 1.12+
        self._any_vec = jl.seval("(args...) -> Any[args...]")
        # {} type application is not expressible in Python syntax
        self._objective_attr = jl.seval("f -> MOI.ObjectiveFunction{typeof(f)}()")
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
        return self._moi.scalar_nonlinear(jl.Symbol(head), self._any_vec(*args))

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
        return self._moi.simplify(func)

    # -- Model building ----------------------------------------------------------

    def add_variables(self, count):
        start = len(self._variables)
        self._variables.extend(self._jl.MOI.add_variables(self._optimizer, count))
        return start

    def add_constraint(self, func, sense, rhs):
        if self._constructors is None:
            self._moi.normalize_and_add_constraint(
                self._optimizer, self._simplify(func), _SENSE_CODES[sense], float(rhs),
            )
        else:
            with self._constructors.scalar_set(sense, rhs) as set_:
                self._jl.MOI.Utilities.normalize_and_add_constraint(
                    self._optimizer, self._simplify(func), set_.to_julia(),
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
        if self._constructors is None:
            set_ = self._moi.vector_set(_SENSE_CODES[sense], n)
            jl.MOI.add_constraint(self._optimizer, generator, set_)
        else:
            with self._constructors.vector_set(sense, n) as set_:
                jl.MOI.add_constraint(self._optimizer, generator, set_.to_julia())

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
