"""
Bridge between JuMPy's Python expression graph and MathOptInterface via juliacall.

Translates Python Expr nodes into Julia MOI + GenOpt types.
This module is only imported when Julia is actually needed (juliacall solves
and MOF export).

The Python side does NO iteration over constraints or variables.
It builds one expression template per constraint group, hands it to GenOpt
as a FunctionGenerator, and lets Julia handle all expansion.

Solver extensions (e.g. the VRP support in jumpy.vrp) plug in through the
SolverFunction / VectorSet protocols defined in jumpy.expressions: the bridge
calls `setup_julia` once per type and `to_moi` per instance, passing an
MOIContext so extensions never touch bridge internals. The bridge itself
contains no knowledge of any concrete extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model

from jumpy.expressions import (
    BinaryOp,
    Constant,
    Func,
    IndexedParameter,
    IndexedVariable,
    SolverFunction,
    UnaryOp,
    Variable,
)
from jumpy.iterators import Iterator


_HELPERS_DEFINED = False
_any_vec = None
_SETUP_TYPES: set[type] = set()


def ensure_package(jl, name: str, url: str | None = None) -> None:
    """Install (if missing) and import a Julia package into Main."""
    add_expr = f'Pkg.add(url = "{url}")' if url else f'Pkg.add("{name}")'
    jl.seval("using Pkg")
    jl.seval(f"""
        if !haskey(Pkg.project().dependencies, "{name}")
            {add_expr}
        end
    """)
    jl.seval(f"import {name}")


def _ensure_setup(jl, cls: type) -> None:
    """Run a SolverFunction / VectorSet subclass's one-time Julia setup."""
    if cls not in _SETUP_TYPES:
        cls.setup_julia(jl)
        _SETUP_TYPES.add(cls)


def _define_helpers(jl):
    """Define Julia helper functions once."""
    global _HELPERS_DEFINED, _any_vec
    if _HELPERS_DEFINED:
        return
    # jl.Any[...] is broken in PythonCall with Julia 1.12+
    _any_vec = jl.seval("(args...) -> Any[args...]")
    jl.seval("""
        function _jumpy_scalar_affine(terms, constant)
            return MOI.ScalarAffineFunction(terms, constant)
        end

        function _jumpy_affine_term(coef, var)
            return MOI.ScalarAffineTerm(coef, var)
        end

        function _jumpy_vaf(terms, constants)
            return MOI.VectorAffineFunction(terms, constants)
        end

        function _jumpy_vaf_term(row, coef, var)
            return MOI.VectorAffineTerm(row, MOI.ScalarAffineTerm(coef, var))
        end

        function _jumpy_set_objective!(optimizer, sense, func)
            MOI.set(optimizer, MOI.ObjectiveSense(), sense)
            F = typeof(func)
            MOI.set(optimizer, MOI.ObjectiveFunction{F}(), func)
        end
    """)
    jl.seval("""
        function _jumpy_add_variables!(optimizer, count, lower, upper, binary, integer)
            vars = MOI.add_variables(optimizer, count)
            if !isnothing(lower)
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.GreaterThan(lower))
                end
            end
            if !isnothing(upper)
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.LessThan(upper))
                end
            end
            if binary
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.ZeroOne())
                end
            elseif integer
                for v in vars
                    MOI.add_constraint(optimizer, v, MOI.Integer())
                end
            end
            return vars
        end

        function _jumpy_add_constraint_group!(optimizer, func, sense)
            n = prod(length.(func.iterators))
            if sense == "<="
                set = MOI.Nonpositives(n)
            elseif sense == ">="
                set = MOI.Nonnegatives(n)
            elseif sense == "=="
                set = MOI.Zeros(n)
            else
                error("Unknown sense: $sense")
            end
            MOI.add_constraint(optimizer, func, set)
        end

        function _jumpy_make_generator(func, iterators, target_type)
            return GenOpt.FunctionGenerator{target_type}(func, iterators)
        end

        function _jumpy_create_optimizer()
            optimizer = MOI.instantiate(
                MOI.OptimizerWithAttributes(HiGHS.Optimizer, "output_flag" => false),
                with_bridge_type = Float64,
            )
            MOI.Bridges.add_bridge(optimizer, GenOpt.FunctionGeneratorBridge{Float64})
            return optimizer
        end

        function _jumpy_get_solution(optimizer, vars)
            return [MOI.get(optimizer, MOI.VariablePrimal(), v) for v in vars]
        end

        function _jumpy_set_var_name!(model, var, name)
            MOI.set(model, MOI.VariableName(), var, name)
            return nothing
        end

        function _jumpy_create_mof_model()
            # Build into a plain Utilities.Model wrapped in bridges. Because a
            # Utilities.Model does NOT natively support GenOpt.FunctionGenerator
            # constraints.
            # (A FileFormats MOF model uses a UniversalFallback that accepts everything,
            # which would suppress bridging and leave the groups unexpanded.)
            inner = MOI.Utilities.Model{Float64}()
            bridged = MOI.Bridges.full_bridge_optimizer(inner, Float64)
            MOI.Bridges.add_bridge(bridged, GenOpt.FunctionGeneratorBridge{Float64})
            return bridged
        end

        function _jumpy_write_mof!(bridged, filename, set_constraints)
            # `bridged.model` is the inner Utilities.Model holding the fully
            # expanded problem, copy it into a MathOptFormat writer and dump it.
            dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_MOF)
            index_map = MOI.copy_to(dest, bridged.model)
            # Custom vector sets (e.g. MathOptVRP.Partition) are not supported
            # by the plain Utilities.Model / bridge stack, so they are not part
            # of the expanded problem above. Add them straight onto the
            # FileFormats model, which accepts arbitrary sets, remapping the
            # variables through the copy_to index map.
            for (vars, set) in set_constraints
                mapped = [index_map[v] for v in vars]
                MOI.add_constraint(dest, MOI.VectorOfVariables(mapped), set)
            end
            MOI.write_to_file(dest, filename)
            return nothing
        end
    """)
    _HELPERS_DEFINED = True


class MOIContext:
    """
    Conversion context handed to SolverFunction.to_moi implementations.

    Wraps the juliacall handle and the Python -> Julia variable mapping, and
    provides small builders so extensions never depend on bridge internals.
    """

    def __init__(self, jl, model: Model):
        self.jl = jl
        self.model = model
        self.jl_var_blocks = []  # one Julia MOI.VariableIndex vector per VariableBlock

    def variable(self, var: Variable):
        """Julia MOI.VariableIndex for a Python Variable."""
        return self.variable_by_index(var.index)

    def variable_by_index(self, index: int):
        offset = 0
        for block, jl_vars in zip(self.model._var_blocks, self.jl_var_blocks):
            if index < offset + block.count:
                return jl_vars[index - offset]  # PythonCall uses 0-based indexing
            offset += block.count
        raise IndexError(f"Variable index {index} out of range")

    def convert(self, expr):
        """Convert a concrete JuMPy Expr into a Julia MOI function."""
        return _expr_to_moi(self, expr)

    def nonlinear(self, head: str, *args):
        """MOI.ScalarNonlinearFunction(head, [args...])."""
        return self.jl.MOI.ScalarNonlinearFunction(self.jl.Symbol(head), _any_vec(*args))

    def matrix(self, rows):
        """Dense Julia Matrix from nested Python lists (row-major)."""
        jl = self.jl
        return jl.seval("(rows...) -> hcat(rows...)'")(
            *[jl.seval("collect")(list(row)) for row in rows]
        )

    def vector_affine(self, entries):
        """
        MOI.VectorAffineFunction from a mixed sequence of numbers and Variables.

        Numbers become constant rows; Variables become unit affine terms.
        """
        jl = self.jl
        terms = jl.seval("MOI.VectorAffineTerm{Float64}[]")
        constants = []
        for i, elem in enumerate(entries):
            if isinstance(elem, Variable):
                jl.push_b(terms, jl._jumpy_vaf_term(i + 1, 1.0, self.variable(elem)))
                constants.append(0.0)
            else:
                constants.append(float(elem))
        jl_constants = jl.seval("c -> collect(Float64, c)")(constants)
        return jl._jumpy_vaf(terms, jl_constants)


def _expand_concrete(expr):
    """
    Rewrite an Expr into a concrete tree with no iterators, sums, or indexing.

    `sum_over(...)` (SumExpr) becomes nested additions, an Iterator becomes a
    Constant, x[expr] (IndexedVariable) becomes its concrete Variable, and
    param[expr] (IndexedParameter) becomes a Constant. This lets the objective
    and individual-constraint converters — which only understand concrete
    trees — handle expressions built with sum_over and indexed access by
    expanding them in Python first. Additions with a literal 0 (e.g. from
    Python's sum() starting at 0) are folded away.

    Constraint *groups* never pass through here: they are expanded in Julia by
    GenOpt, so their iterators stay symbolic.
    """
    from jumpy.model import SumExpr
    from jumpy.mof_export import _eval_scalar # dont forget to migrate this function to remove mof_export file
    from jumpy.expressions import Constant as _Constant

    def is_zero(e):
        return isinstance(e, Constant) and e.value == 0.0

    def go(e, subs):
        match e:
            case Constant() | Variable():
                return e
            case Iterator() as it:
                return _Constant(float(subs[it.id]))
            case IndexedVariable() as iv:
                idx = int(_eval_scalar(iv.index_expr, subs))
                return iv.variable_vector._variables[idx]
            case IndexedParameter() as ip:
                idx = int(_eval_scalar(ip.index_expr, subs))
                return _Constant(float(ip.parameter.values[idx]))
            case SumExpr() as s:
                terms = [go(s.body, {**subs, s.iterator.id: v})
                         for v in s.iterator.values]
                if not terms:
                    return _Constant(0.0)
                acc = terms[0]
                for t in terms[1:]:
                    acc = BinaryOp("+", acc, t)
                return acc
            case BinaryOp(op=op, left=left, right=right):
                l, r = go(left, subs), go(right, subs)
                if op == "+":
                    if is_zero(l):
                        return r
                    if is_zero(r):
                        return l
                return BinaryOp(op, l, r)
            case UnaryOp(op=op, arg=arg):
                return UnaryOp(op, go(arg, subs))
            case Func(name=name, arg=arg):
                return Func(name, go(arg, subs))
            case _:
                return e

    return go(expr, {})


def _populate_optimizer(jl, optimizer, model: Model) -> MOIContext:
    """
    Build the JuMPy model into a Julia MOI optimizer/model.

    Adds variables, constraint groups (as GenOpt.FunctionGenerator objects so
    expansion happens in Julia), individual constraints, and the objective.
    Returns the MOIContext holding the per-block Julia variable vectors.
    """
    ctx = MOIContext(jl, model)

    # Add variables — one bulk call per block
    for block in model._var_blocks:
        lower = float(block.lower) if block.lower is not None else jl.nothing
        upper = float(block.upper) if block.upper is not None else jl.nothing
        block_vars = jl._jumpy_add_variables_b(
            optimizer, block.count, lower, upper,
            block.binary,
            block.integer,
        )
        ctx.jl_var_blocks.append(block_vars)

    # Add constraint groups via GenOpt
    for group in model._constraint_groups:
        _add_constraint_group(ctx, optimizer, group)

    # Add individual constraints
    for con in model._individual_constraints:
        _add_individual_constraint(ctx, optimizer, con)

    # Set constraints (`variables in set`) are intentionally NOT added here:
    # the plain Utilities.Model / bridge stack does not support custom vector
    # sets. write_mof attaches them directly to the FileFormats model; solve
    # backends that support them (e.g. Vroom) read them off the Model.

    # Set objective
    if model._objective is not None:
        sense = (
            jl.MOI.MIN_SENSE
            if model._objective.sense == "min"
            else jl.MOI.MAX_SENSE
        )
        obj_func = _expr_to_moi(ctx, _expand_concrete(model._objective.expr)) # expand concrete return None dans beaucoup de cas
        jl._jumpy_set_objective_b(optimizer, sense, obj_func)

    return ctx


def build_moi_model(jl, model: Model) -> list[float]:
    """
    Build an MOI model in Julia from a JuMPy Model and solve it.

    Constraint groups are passed as GenOpt.FunctionGenerator objects
    so that expansion happens entirely in Julia.
    """
    _define_helpers(jl)

    optimizer = jl._jumpy_create_optimizer()
    ctx = _populate_optimizer(jl, optimizer, model)

    # Optimize and extract solution
    jl.MOI.optimize_b(optimizer)

    # Flatten all variable blocks into one solution vector
    all_vars_flat = jl.seval("vcat")(*ctx.jl_var_blocks)
    jl_solution = jl._jumpy_get_solution(optimizer, all_vars_flat)
    return [float(jl_solution[i]) for i in range(len(jl_solution))]


def _is_linear_template(expr) -> bool:
    """Check if a template expression is linear (no nonlinear functions)."""
    match expr:
        case Constant() | Variable() | Iterator() | IndexedVariable() | IndexedParameter():
            return True
        case BinaryOp(op=op, left=left, right=right):
            if op in ("+", "-", "*"):
                return _is_linear_template(left) and _is_linear_template(right)
            return False
        case UnaryOp(op="-", arg=arg):
            return _is_linear_template(arg)
        case _:
            return False


def _add_constraint_group(ctx: MOIContext, optimizer, group):
    """
    Add a constraint group as a single GenOpt.FunctionGenerator.

    Python builds the template expression and iterator list, then hands
    them to GenOpt. No Python-side iteration over constraint instances.
    """
    jl = ctx.jl

    # Build GenOpt iterators
    genopt_iterators = jl.seval("GenOpt.Iterator[]")
    iter_id_map = {}
    for idx, it in enumerate(group.iterators):
        jl_values = jl.seval("collect")(it.values)
        jl_it = jl.GenOpt.Iterator(jl_values)
        jl.push_b(genopt_iterators, jl_it)
        iter_id_map[it.id] = idx + 1  # 1-based

    # Normalize: lhs - rhs in {Nonpositives, Nonnegatives, Zeros}
    normalized = group.template.lhs - group.template.rhs

    # Build MOI.ScalarNonlinearFunction template with GenOpt placeholders
    template_func = _expr_to_moi_template(ctx, normalized, iter_id_map)

    # Determine target function type: affine if template is linear, else nonlinear
    if _is_linear_template(group.template.lhs) and _is_linear_template(group.template.rhs):
        target_type = jl.seval("MOI.ScalarAffineFunction{Float64}")
    else:
        target_type = jl.seval("MOI.ScalarNonlinearFunction")

    # Wrap in FunctionGenerator and add constraint — all in Julia
    func_gen = jl._jumpy_make_generator(template_func, genopt_iterators, target_type)
    jl._jumpy_add_constraint_group_b(optimizer, func_gen, group.template.sense)


def _get_contiguous(jl, variable_vector):
    """Get a GenOpt.ContiguousArrayOfVariables for a VariableVector."""
    start = variable_vector._variables[0].index
    count = len(variable_vector)
    return jl.seval(
        f"GenOpt.ContiguousArrayOfVariables({start}, ({count},))"
    )


def _expr_to_moi_template(ctx: MOIContext, expr, iter_id_map):
    """
    Convert a Python Expr into an MOI.ScalarNonlinearFunction template
    with GenOpt.IteratorIndex and ContiguousArrayOfVariables placeholders.
    """
    jl = ctx.jl
    match expr:
        case Constant(value=v):
            return v
        case Variable():
            return ctx.variable(expr)
        case SolverFunction():
            raise TypeError(
                f"{type(expr).__name__} is not supported inside constraint "
                "group templates"
            )
        case BinaryOp(op=op, left=left, right=right):
            l = _expr_to_moi_template(ctx, left, iter_id_map)
            r = _expr_to_moi_template(ctx, right, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), _any_vec(l, r))
        case UnaryOp(op="-", arg=arg):
            a = _expr_to_moi_template(ctx, arg, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), _any_vec(a))
        case Func(name=name, arg=arg):
            a = _expr_to_moi_template(ctx, arg, iter_id_map)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), _any_vec(a))
        case Iterator() as it:
            return jl.GenOpt.IteratorIndex(iter_id_map[it.id])
        case IndexedVariable() as iv:
            contiguous = _get_contiguous(jl, iv.variable_vector)
            index_expr = _expr_to_moi_template(ctx, iv.index_expr, iter_id_map)
            # 0-based Python → 1-based Julia
            index_1based = jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("+"), _any_vec(index_expr, 1),
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), _any_vec(contiguous, index_1based),
            )
        case IndexedParameter() as ip:
            jl_values = jl.seval("collect")(ip.parameter.values)
            index_expr = _expr_to_moi_template(ctx, ip.index_expr, iter_id_map)
            index_1based = jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("+"), _any_vec(index_expr, 1),
            )
            return jl.MOI.ScalarNonlinearFunction(
                jl.Symbol("getindex"), _any_vec(jl_values, index_1based),
            )
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI template")


def _collect_linear_terms(expr, terms, sign=1.0):
    """
    Try to decompose expr into linear terms: list of (coef, var_index) + constant.
    Returns (success, constant).
    """
    match expr:
        case Constant(value=v):
            return True, v * sign
        case Variable(index=idx):
            terms.append((sign, idx))
            return True, 0.0
        case BinaryOp(op="+", left=left, right=right):
            terms_before = len(terms)
            ok_l, const_l = _collect_linear_terms(left, terms, sign)
            if not ok_l:
                del terms[terms_before:]
                return False, 0.0
            ok_r, const_r = _collect_linear_terms(right, terms, sign)
            if not ok_r:
                del terms[terms_before:]
                return False, 0.0
            return True, const_l + const_r
        case BinaryOp(op="-", left=left, right=right):
            terms_before = len(terms)
            ok_l, const_l = _collect_linear_terms(left, terms, sign)
            if not ok_l:
                del terms[terms_before:]
                return False, 0.0
            ok_r, const_r = _collect_linear_terms(right, terms, -sign)
            if not ok_r:
                del terms[terms_before:]
                return False, 0.0
            return True, const_l + const_r
        case BinaryOp(op="*", left=Constant(value=v), right=right):
            return _collect_linear_terms(right, terms, sign * v)
        case BinaryOp(op="*", left=left, right=Constant(value=v)):
            return _collect_linear_terms(left, terms, sign * v)
        case UnaryOp(op="-", arg=arg):
            return _collect_linear_terms(arg, terms, -sign)
        case _:
            return False, 0.0


def _expr_to_moi_linear(ctx: MOIContext, expr):
    """
    Try to convert expr to ScalarAffineFunction. Returns None if nonlinear.
    """
    jl = ctx.jl
    terms = []
    ok, constant = _collect_linear_terms(expr, terms)
    if not ok:
        return None

    # Aggregate repeated variables into a single term and drop zero
    # coefficients, so the affine function is in canonical form (e.g.
    # x + x + x -> 3 x, and x - x -> dropped).
    agg = {}
    order = []
    for coef, var_idx in terms:
        if var_idx not in agg:
            order.append(var_idx)
            agg[var_idx] = 0.0
        agg[var_idx] += coef

    jl_terms = jl.seval("MOI.ScalarAffineTerm{Float64}[]")
    for var_idx in order:
        coef = agg[var_idx]
        if coef == 0.0:
            continue
        jl_var = ctx.variable_by_index(var_idx)
        jl.push_b(jl_terms, jl._jumpy_affine_term(float(coef), jl_var))

    return jl._jumpy_scalar_affine(jl_terms, float(constant))


def _expr_to_moi(ctx: MOIContext, expr):
    """Convert a Python Expr to a concrete Julia MOI function (no iterators)."""
    # Try linear first
    linear = _expr_to_moi_linear(ctx, expr)
    if linear is not None:
        return linear

    jl = ctx.jl
    match expr:
        case Constant(value=v):
            return v
        case Variable():
            return ctx.variable(expr)
        case SolverFunction():
            _ensure_setup(jl, type(expr))
            return expr.to_moi(ctx)
        case BinaryOp(op=op, left=left, right=right):
            l = _moi_arg(ctx, left)
            r = _moi_arg(ctx, right)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(op), _any_vec(l, r))
        case UnaryOp(op="-", arg=arg):
            a = _moi_arg(ctx, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol("-"), _any_vec(a))
        case Func(name=name, arg=arg):
            a = _moi_arg(ctx, arg)
            return jl.MOI.ScalarNonlinearFunction(jl.Symbol(name), _any_vec(a))
        case _:
            raise TypeError(f"Cannot convert {type(expr).__name__} to MOI")


def _moi_arg(ctx: MOIContext, expr):
    """
    Convert a sub-expression of a nonlinear node.

    Bare variables and constants stay plain MOI leaves (VariableIndex / number)
    instead of being wrapped in a 1-term ScalarAffineFunction, which keeps the
    nonlinear tree — and its MOF serialization — minimal.
    """
    match expr:
        case Constant(value=v):
            return v
        case Variable():
            return ctx.variable(expr)
        case _:
            return _expr_to_moi(ctx, expr)


def _add_individual_constraint(ctx: MOIContext, optimizer, con):
    """Add a single non-grouped constraint."""
    jl = ctx.jl
    # Normalize: lhs - rhs in {set}
    normalized = _expand_concrete(con.lhs - con.rhs)

    func = _expr_to_moi(ctx, normalized)

    if con.sense == "<=":
        set_ = jl.MOI.LessThan(0.0)
    elif con.sense == ">=":
        set_ = jl.MOI.GreaterThan(0.0)
    elif con.sense == "==":
        set_ = jl.MOI.EqualTo(0.0)
    else:
        raise ValueError(f"Unknown constraint sense: {con.sense}")

    jl.MOI.Utilities.normalize_and_add_constraint(optimizer, func, set_)


def _jl_set_constraints(ctx: MOIContext):
    """
    Convert the model's set constraints into (variables, MOI set) pairs for
    the MOF writer, dispatching to each VectorSet's to_moi / setup_julia.
    """
    jl = ctx.jl
    pairs = jl.seval("Tuple{Vector{MOI.VariableIndex},Any}[]")
    make_pair = jl.seval("(v, s) -> (v, s)")
    for con in ctx.model._set_constraints:
        _ensure_setup(jl, type(con.set_))
        jl_vars = jl.seval("MOI.VariableIndex[]")
        for var in con.variables:
            jl.push_b(jl_vars, ctx.variable(var))
        jl.push_b(pairs, make_pair(jl_vars, con.set_.to_moi(jl)))
    return pairs


_mof_backend = None


def write_mof(model: Model, filename) -> None:
    """
    Write `model` to a MathOptFormat (.mof.json) file at `filename`.

    Builds the model in Julia through MOI exactly like the juliacall solve
    path, but targets a `MOI.FileFormats` MOF model instead of a solver.
    Constraint groups are expanded by the GenOpt bridge on the Julia side, so
    the resulting file contains fully instantiated constraints. Set constraints
    and solver-specific functions are serialized through the VectorSet /
    SolverFunction protocols, so any model — VRP or not — exports the same way,
    regardless of which backend it was created with.
    """
    # MOF export always goes through juliacall, independent of the model's
    # solve backend (juliacall.Main is process-global, so this shares the
    # Julia session with any juliacall-based backend).
    global _mof_backend
    if _mof_backend is None:
        from jumpy.backend import JuliaCallBackend
        _mof_backend = JuliaCallBackend()
    _mof_backend._init_julia()
    jl = _mof_backend._jl

    _define_helpers(jl)

    dest = jl._jumpy_create_mof_model()  # create the bridged model
    ctx = _populate_optimizer(jl, dest, model)

    # Preserve JuMPy variable names in the MOF output (x[0], route[2], or the
    # v{index} fallback for unnamed variables).
    for block, jl_vars in zip(model._var_blocks, ctx.jl_var_blocks):
        for k in range(block.count):
            var = block.vector._variables[k]
            name = var.name if var.name else f"v{var.index}"
            jl._jumpy_set_var_name_b(dest, jl_vars[k], name)

    # Vector sets may impose their own naming convention on the variables they
    # constrain (e.g. Partition's 2-D 1-based `nodes[i,t]` layout).
    for con in model._set_constraints:
        names = con.set_.variable_names(con.variables)
        if names is None:
            continue
        for var, name in zip(con.variables, names):
            jl._jumpy_set_var_name_b(dest, ctx.variable(var), name)

    # Set constraints can't live in the bridged model; they are added to the
    # FileFormats model after copy_to inside _jumpy_write_mof!.
    jl._jumpy_write_mof_b(dest, str(filename), _jl_set_constraints(ctx))