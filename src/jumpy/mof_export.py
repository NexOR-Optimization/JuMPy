import json
from itertools import product
from collections import defaultdict

"""
Script intended to export optimization problem as mof files
(Note that quadratic expressions are being handled as nonlinear expression)
"""

from jumpy.expressions import (
    BinaryOp,
    Constant,
    Func,
    IndexedParameter,
    IndexedVariable,
    UnaryOp,
    Variable,
)
from jumpy.iterators import Iterator


def write_mof(model, filename):
    """Write a JuMPy model to a MathOptFormat .mof.json file """
    with open(filename, "w") as f:
        json.dump(_model_to_mof(model), f, indent=2)


def _model_to_mof(model):
    variables, var_constraints = _emit_variables(model)
    return {
        "name": "JuMPy Model",
        "version": {"major": 1, "minor": 7},
        "variables": variables,
        "objective": _emit_objective(model),
        "constraints": var_constraints + _emit_constraints(model),
    }


# --- Variable naming ---------------------------------------------------------

def _var_name(var):
    """MOF name for a Variable. Falls back to v{index} if unnamed."""
    return var.name if var.name else f"v{var.index}"


# --- Variables and bounds ----------------------------------------------------

def _emit_variables(model):
    variables, bounds = [], []
    for block in model._var_blocks:
        for k in range(block.count):
            var = block.vector._variables[k]
            vname = _var_name(var)
            variables.append({"name": vname})
            if block.lower is not None:
                bounds.append(_var_bound(vname, "GreaterThan", "lower", block.lower))
            if block.upper is not None:
                bounds.append(_var_bound(vname, "LessThan", "upper", block.upper))
            if block.binary:
                bounds.append({"function": {"type": "Variable", "name": vname},
                               "set": {"type": "ZeroOne"}})
            elif block.integer:
                bounds.append({"function": {"type": "Variable", "name": vname},
                               "set": {"type": "Integer"}})

    return variables, bounds


def _var_bound(vname, set_type, set_key, value):
    return {"function": {"type": "Variable", "name": vname},
            "set": {"type": set_type, set_key: float(value)}}


# --- Objective ---------------------------------------------------------------

def _emit_objective(model):
    obj = model._objective
    if obj is None:
        return {"sense": "feasibility"}
    return {"sense": "min" if obj.sense == "min" else "max",
            "function": _expr_to_function_any(obj.expr, {})}


# --- Constraints -------------------------------------------------------------

def _emit_constraints(model):
    out = []
    for c in model._individual_constraints:
        out.append(_constraint_to_mof(c, {}, name=None))
    for gi, g in enumerate(model._constraint_groups):
        ranges = [list(it.values) for it in g.iterators]
        for ci, combo in enumerate(product(*ranges)):
            subs = {it.id: val for it, val in zip(g.iterators, combo)}
            out.append(_constraint_to_mof(g.template, subs, f"cg{gi}_{ci}"))
    return out


_SET_MAP = {"<=": ("LessThan", "upper"),
            ">=": ("GreaterThan", "lower"),
            "==": ("EqualTo", "value")}

def _constraint_to_mof(con, subs, name):
    normalized = BinaryOp("-", con.lhs, con.rhs)
    set_type, key = _SET_MAP[con.sense]

    try:
        func = _expr_to_function(normalized, subs)
    except (NotImplementedError, _NotScalar):
        # Nonlinear / bilinear constraint. A ScalarNonlinearFunction has no
        # extractable "constant" field, so we keep the whole (lhs - rhs)
        # expression as the function and constrain it against 0:
        #     h(x) {<=, >=, ==} 0
        func = _expr_to_nonlinear_function(normalized, subs)
        bound = 0.0
    else:
        # Affine: fold the function's constant into the set's bound.
        # g(x) + c {<=, >=, ==} 0  ->  g(x) {<=, >=, ==} -c
        c = func["constant"]
        func["constant"] = 0.0
        bound = -c

    bound = float(bound)
    if bound == 0.0:
        bound = 0.0  # normalize -0.0 -> 0.0
    entry = {"function": func, "set": {"type": set_type, key: bound}}
    if name:
        entry["name"] = name
    return entry

# --- Function emission: affine when possible, else nonlinear -----------------

def _expr_to_function_any(expr, subs):
    """
    Emit the simplest MOF function that faithfully represents `expr`.

    Linear expressions become a ScalarAffineFunction (the canonical, most
    widely supported form). Anything the affine walker can't express
    (nonlinear functions like sin/exp, products of variables, powers,
    division by a variable, ...) falls back to a ScalarNonlinearFunction.

    Note that implementing quadratic could be nice
    """
    try:
        return _expr_to_function(expr, subs)
    except (NotImplementedError, _NotScalar):
        return _expr_to_nonlinear_function(expr, subs)


# --- Affine walker -----------------------------------------------------------

def _expr_to_function(expr, subs):
    terms, constant = _flatten_affine(expr, subs)
    return {
        "type": "ScalarAffineFunction",
        "terms": [{"coefficient": float(c), "variable": vname}
                  for vname, c in terms.items() if c != 0.0],
        "constant": float(constant),
    }


def _flatten_affine(expr, subs):
    """Walk an expression, return ({var_name: coef}, constant)"""
    terms = defaultdict(float)
    const_box = [0.0]
    _walk(expr, 1.0, terms, const_box, subs)
    return dict(terms), const_box[0]


def _walk(e, sign, terms, const_box, subs):
    # Lazy import to break the model <-> mof_export circular dependency
    from jumpy.model import SumExpr

    if isinstance(e, (int, float)):
        const_box[0] += sign * float(e)
        return
    if isinstance(e, Constant):
        const_box[0] += sign * float(e.value)
        return
    if isinstance(e, Variable):
        terms[_var_name(e)] += sign
        return
    if isinstance(e, IndexedVariable):
        terms[_resolve_indexed_var(e, subs)] += sign
        return
    if isinstance(e, IndexedParameter):
        const_box[0] += sign * _resolve_indexed_param(e, subs)
        return
    if isinstance(e, Iterator):
        const_box[0] += sign * float(subs[e.id])
        return
    if isinstance(e, SumExpr):
        for val in e.iterator.values:
            local_subs = {**subs, e.iterator.id: val}
            _walk(e.body, sign, terms, const_box, local_subs)
        return
    if isinstance(e, UnaryOp):
        if e.op == "-":
            _walk(e.arg, -sign, terms, const_box, subs)
            return
        raise NotImplementedError(f"Unary op '{e.op}' not supported")
    if isinstance(e, BinaryOp):
        if e.op == "+":
            _walk(e.left, sign, terms, const_box, subs)
            _walk(e.right, sign, terms, const_box, subs)
            return
        if e.op == "-":
            _walk(e.left, sign, terms, const_box, subs)
            _walk(e.right, -sign, terms, const_box, subs)
            return
        if e.op == "*":
            _walk_mul(e.left, e.right, sign, terms, const_box, subs)
            return
        if e.op == "/":
            divisor = _eval_scalar(e.right, subs)
            if divisor == 0:
                raise ZeroDivisionError("Division by zero in expression")
            _walk(e.left, sign / divisor, terms, const_box, subs)
            return
        raise NotImplementedError(f"Binary op '{e.op}' is nonlinear or unsupported")
    if isinstance(e, Func):
        raise NotImplementedError(f"Nonlinear function '{e.name}' cannot be emitted as affine")
    raise NotImplementedError(f"Unknown expression type: {type(e).__name__}")


def _walk_mul(left, right, sign, terms, const_box, subs):
    """Handle a*b where (at least) one side must be a scalar (affine restriction)."""
    try:
        scalar = _eval_scalar(left, subs)
        _walk(right, sign * scalar, terms, const_box, subs)
        return
    except _NotScalar:
        pass
    try:
        scalar = _eval_scalar(right, subs)
        _walk(left, sign * scalar, terms, const_box, subs)
        return
    except _NotScalar:
        pass
    # Signals the caller to fall back to the nonlinear emitter.
    raise NotImplementedError(
        "Bilinear/quadratic term encountered (var * var); not affine"
    )


class _NotScalar(Exception):
    """Raised when an expression can't be evaluated to a numeric scalar."""


def _eval_scalar(e, subs):
    """Evaluate an expression as a scalar (no variables). Raises _NotScalar otherwise."""
    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, Constant):
        return float(e.value)
    if isinstance(e, Iterator):
        if e.id not in subs:
            raise _NotScalar(f"Unbound iterator {e}")
        return float(subs[e.id])
    if isinstance(e, IndexedParameter):
        return _resolve_indexed_param(e, subs)
    if isinstance(e, UnaryOp):
        if e.op == "-":
            return -_eval_scalar(e.arg, subs)
        raise _NotScalar(f"Unary op '{e.op}' not scalar-evaluable")
    if isinstance(e, BinaryOp):
        l = _eval_scalar(e.left, subs)
        r = _eval_scalar(e.right, subs)
        if e.op == "+": return l + r
        if e.op == "-": return l - r
        if e.op == "*": return l * r
        if e.op == "/": return l / r
        if e.op == "^": return l ** r
        raise _NotScalar(f"Unknown binary op '{e.op}'")
    # Variable, IndexedVariable, Func, SumExpr → not scalars
    raise _NotScalar(f"{type(e).__name__} cannot be evaluated as scalar")


# --- Nonlinear walker --------------------------------------------------------
#
# Emits a MOF "ScalarNonlinearFunction":
#
#     {"type": "ScalarNonlinearFunction", "root": <node>, "node_list": [<node>...]}
#
# The expression graph is stored in prefix form. Operator sub-nodes are
# flattened into `node_list` and referenced from their parent's `args` by a
# 1-based index via {"type": "node", "index": k}. Leaf nodes are inlined: a
# variable is a bare string (its name), a numeric constant is a bare number.

# JuMPy's Func names already coincide with MOI's registered nonlinear
# operators, so this mapping is essentially identity; it is kept explicit so
# future divergences are a one-line change.
_MOF_FUNC = {
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
}


def _expr_to_nonlinear_function(expr, subs):
    """Emit a MOF ScalarNonlinearFunction for an arbitrary expression."""
    node_list = []
    root = _build_node(expr, subs, node_list)
    return {
        "type": "ScalarNonlinearFunction",
        "root": root,
        "node_list": node_list,
    }


def _arg_ref(e, subs, node_list):
    """
    Build `e` and return a value usable inside a parent node's `args` list.

    Operator nodes are appended to `node_list` and replaced by a 1-based
    {"type": "node", "index": k} reference; leaf nodes (str / float) are
    returned inline.
    """
    node = _build_node(e, subs, node_list)
    if isinstance(node, dict):  # an operator node (always has "args")
        node_list.append(node)
        return {"type": "node", "index": len(node_list)}  # 1-based
    return node


def _build_node(e, subs, node_list):
    """
    Return a MOF nonlinear node for `e`:
        - a bare str   for a variable leaf,
        - a bare float for a numeric leaf,
        - a dict {"type": op, "args": [...]} for an operator node.
    """
    from jumpy.model import SumExpr

    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, Constant):
        return float(e.value)
    if isinstance(e, Variable):
        return _var_name(e)
    if isinstance(e, IndexedVariable):
        return _resolve_indexed_var(e, subs)
    if isinstance(e, IndexedParameter):
        return _resolve_indexed_param(e, subs)
    if isinstance(e, Iterator):
        if e.id not in subs:
            raise NotImplementedError(f"Unbound iterator {e} outside a sum/constraint group")
        return float(subs[e.id])
    if isinstance(e, SumExpr):
        args = [_arg_ref(e.body, {**subs, e.iterator.id: val}, node_list)
                for val in e.iterator.values]
        if not args:
            return 0.0
        return {"type": "+", "args": args}
    if isinstance(e, UnaryOp):
        if e.op == "-":
            return {"type": "-", "args": [_arg_ref(e.arg, subs, node_list)]}
        raise NotImplementedError(f"Unary op '{e.op}' not supported")
    if isinstance(e, BinaryOp):
        if e.op in ("+", "-", "*", "/", "^"):
            return {"type": e.op,
                    "args": [_arg_ref(e.left, subs, node_list),
                             _arg_ref(e.right, subs, node_list)]}
        raise NotImplementedError(f"Binary op '{e.op}' not supported")
    if isinstance(e, Func):
        op = _MOF_FUNC.get(e.name)
        if op is None:
            raise NotImplementedError(f"Function '{e.name}' has no MOF mapping")
        return {"type": op, "args": [_arg_ref(e.arg, subs, node_list)]}
    raise NotImplementedError(f"Cannot emit nonlinear node for {type(e).__name__}")


# --- Index resolution --------------------------------------------------------

def _resolve_indexed_var(iv, subs):
    """Resolve x[expr] to a concrete variable name given current iterator subs."""
    idx = int(_eval_scalar(iv.index_expr, subs))
    var = iv.variable_vector._variables[idx]
    return _var_name(var)


def _resolve_indexed_param(ip, subs):
    """Resolve param[expr] to a concrete numeric value."""
    idx = int(_eval_scalar(ip.index_expr, subs))
    return float(ip.parameter.values[idx])