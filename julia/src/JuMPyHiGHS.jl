module JuMPyHiGHS

# C entry points for the JuMPy juliac backend.
#
# The ABI mirrors the MOI API one function per entry point, so that the
# Python code building the model is the same for the juliacall and juliac
# backends: `jumpy_scalar_nonlinear` is the compiled counterpart of
# `jl.MOI.ScalarNonlinearFunction(...)`, `jumpy_add_constraint` of
# `MOI.Utilities.normalize_and_add_constraint(...)`, and so on. Nothing here
# is HiGHS-specific except the `Optimizer` constant below: compiling any
# other MOI optimizer behind the same entry points is a one-line change.
#
# The optimizer is used raw: no `MOI.Bridges`, no
# `MOI.Utilities.CachingOptimizer`. GenOpt is compiled in and
# jumpy_add_group_constraint expands templates here. Whatever functions and
# sets the optimizer does not support are reported as errors.
#
# Conventions across the C ABI:
#   - a model is an opaque pointer returned by jumpy_new_model; it stays
#     valid until jumpy_free_model, after which it must not be used
#   - MOI functions are opaque pointers built with jumpy_constant /
#     jumpy_variable / jumpy_scalar_nonlinear; they belong to the model
#     that built them and are freed with it
#   - variables are 0-based column indices in the order they were added
#   - constraint sense: 0 = <=, 1 = >=, 2 = ==, 3 = binary, 4 = integer
#   - objective sense: 0 = min, 1 = max
#   - entry points return -1 (NULL, NaN) on error, after printing to stderr
#     (a Julia exception must never propagate across the C boundary)

import GenOpt
import HiGHS
import MathOptInterface as MOI

# The only solver-specific line in this package.
const Optimizer = HiGHS.Optimizer

mutable struct ModelHandle
    optimizer::Optimizer
    variables::Vector{MOI.VariableIndex}
    # Roots the expression nodes handed out as pointers: the Julia GC
    # cannot see references held by the C caller.
    nodes::Vector{Base.RefValue{Any}}
end

# Same rooting, for the models themselves: alive until jumpy_free_model.
const KEEP_ALIVE = IdDict{ModelHandle,Nothing}()
const LOCK = ReentrantLock()

function _get(model::Ptr{Cvoid})
    model == C_NULL && error("Model pointer is NULL")
    return unsafe_pointer_to_objref(model)::ModelHandle
end

# Expression nodes (Float64, MOI.VariableIndex, MOI functions) are boxed in
# a Ref so that immutable values also get a stable pointer.
function _box(handle::ModelHandle, value)::Ptr{Cvoid}
    node = Base.RefValue{Any}(value)
    push!(handle.nodes, node)
    return pointer_from_objref(node)
end

function _unbox(node::Ptr{Cvoid})
    node == C_NULL && error("Expression pointer is NULL")
    return (unsafe_pointer_to_objref(node)::Base.RefValue{Any})[]
end

macro _catch(default, expr)
    quote
        try
            $(esc(expr))
        catch err
            print(stderr, "JuMPyHiGHS error: ")
            showerror(stderr, err)
            println(stderr)
            $(esc(default))
        end
    end
end

# -- Model lifecycle ----------------------------------------------------------

Base.@ccallable function jumpy_new_model()::Ptr{Cvoid}
    @_catch C_NULL begin
        optimizer = Optimizer()
        MOI.set(optimizer, MOI.Silent(), true)
        handle = ModelHandle(optimizer, MOI.VariableIndex[], Base.RefValue{Any}[])
        Base.@lock LOCK KEEP_ALIVE[handle] = nothing
        pointer_from_objref(handle)
    end
end

Base.@ccallable function jumpy_free_model(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        Base.@lock LOCK delete!(KEEP_ALIVE, _get(model))
        Cint(0)
    end
end

# -- Variables ----------------------------------------------------------------

# MOI.add_variables. Returns the 0-based index of the first added variable.
# Bounds are constraints: pass a variable node to jumpy_add_constraint.
Base.@ccallable function jumpy_add_variables(
    model::Ptr{Cvoid},
    count::Clonglong,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        start = length(handle.variables)
        append!(handle.variables, MOI.add_variables(handle.optimizer, count))
        Clonglong(start)
    end
end

# -- MOI function constructors --------------------------------------------------

Base.@ccallable function jumpy_constant(
    model::Ptr{Cvoid},
    value::Cdouble,
)::Ptr{Cvoid}
    @_catch C_NULL _box(_get(model), value)
end

# MOI.VariableIndex of the 0-based column `index`.
Base.@ccallable function jumpy_variable(
    model::Ptr{Cvoid},
    index::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, handle.variables[index+1])
    end
end

# MOI.ScalarNonlinearFunction(Symbol(head), Any[args...]).
Base.@ccallable function jumpy_scalar_nonlinear(
    model::Ptr{Cvoid},
    head::Cstring,
    args::Ptr{Ptr{Cvoid}},
    nargs::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        func = MOI.ScalarNonlinearFunction(
            Symbol(unsafe_string(head)),
            Any[_unbox(unsafe_load(args, k)) for k in 1:nargs],
        )
        _box(handle, func)
    end
end

# GenOpt.IteratorRef over the given values: a template node usable in
# jumpy_scalar_nonlinear args, expanded by jumpy_add_group_constraint.
Base.@ccallable function jumpy_iterator(
    model::Ptr{Cvoid},
    values::Ptr{Cdouble},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        iterator = GenOpt.Iterator([unsafe_load(values, k) for k in 1:len])
        _box(handle, GenOpt.IteratorRef(iterator))
    end
end

# GenOpt.ContiguousArrayOfVariables: the block of `count` variables starting
# at 0-based column `offset`, indexable (1-based) inside a template.
Base.@ccallable function jumpy_contiguous_variables(
    model::Ptr{Cvoid},
    offset::Clonglong,
    count::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, GenOpt.ContiguousArrayOfVariables(offset, (Int64(count),)))
    end
end

# A data vector, indexable (1-based) inside a template.
Base.@ccallable function jumpy_float_array(
    model::Ptr{Cvoid},
    values::Ptr{Cdouble},
    len::Clonglong,
)::Ptr{Cvoid}
    @_catch C_NULL begin
        handle = _get(model)
        _box(handle, [unsafe_load(values, k) for k in 1:len])
    end
end

# -- Constraints --------------------------------------------------------------

# Simplify returns a ScalarAffineFunction when the expression is affine, so
# optimizers without nonlinear support (like HiGHS) accept it. Never narrow
# below ScalarAffineFunction: `x >= 0` as a constraint must stay a row, not
# become a VariableIndex bound (same semantics as JuMP's @constraint). Only
# a function that already is a VariableIndex — the bounds path — is a bound.
function _simplify(func::MOI.ScalarNonlinearFunction)
    f = MOI.Nonlinear.SymbolicAD.simplify(func)
    if f isa MOI.VariableIndex || f isa Float64
        return convert(MOI.ScalarAffineFunction{Float64}, f)
    end
    return f
end
_simplify(func) = func

function _scalar_set(sense::Cint, rhs::Float64)
    if sense == 0
        return MOI.LessThan(rhs)
    elseif sense == 1
        return MOI.GreaterThan(rhs)
    elseif sense == 2
        return MOI.EqualTo(rhs)
    elseif sense == 3
        return MOI.ZeroOne()
    elseif sense == 4
        return MOI.Integer()
    end
    return error("Invalid constraint sense: $sense")
end

# MOI.add_constraint(func, set) where set is
# {0: LessThan, 1: GreaterThan, 2: EqualTo}(rhs) or {3: ZeroOne, 4: Integer}
# (rhs ignored). Function constants are
# normalized into the set. Returns the raw MOI constraint index value.
Base.@ccallable function jumpy_add_constraint(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
    sense::Cint,
    rhs::Cdouble,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        ci = MOI.Utilities.normalize_and_add_constraint(
            handle.optimizer,
            _simplify(_unbox(func)),
            _scalar_set(sense, rhs),
        )
        Clonglong(ci.value)
    end
end

# Expand a template containing GenOpt.IteratorRef nodes into one scalar
# constraint `expanded(func) sense 0` per combination of iterator values —
# the same expansion loop as GenOpt.FunctionGeneratorBridge, on the raw
# optimizer. Returns the number of constraints added.
Base.@ccallable function jumpy_add_group_constraint(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
    sense::Cint,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        set = _scalar_set(sense, 0.0)
        template, iterators = GenOpt.collect_iterator_refs(
            _unbox(func)::MOI.ScalarNonlinearFunction,
        )
        sizes = Tuple(length.(iterators))
        for idx in CartesianIndices(sizes)
            values = [iterators[k].values[idx[k]] for k in eachindex(iterators)]
            expanded = GenOpt._expand(template, values)
            MOI.Utilities.normalize_and_add_constraint(
                handle.optimizer,
                _simplify(expanded),
                set,
            )
        end
        Clonglong(prod(sizes))
    end
end

# -- Objective ----------------------------------------------------------------

# MOI.set(MOI.ObjectiveSense()); 0 = min, 1 = max.
Base.@ccallable function jumpy_set_objective_sense(
    model::Ptr{Cvoid},
    sense::Cint,
)::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        moi_sense = sense == 0 ? MOI.MIN_SENSE : MOI.MAX_SENSE
        MOI.set(handle.optimizer, MOI.ObjectiveSense(), moi_sense)
        Cint(0)
    end
end

# MOI.set(MOI.ObjectiveFunction{F}(), func).
Base.@ccallable function jumpy_set_objective_function(
    model::Ptr{Cvoid},
    func::Ptr{Cvoid},
)::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        f = _simplify(_unbox(func))
        if f isa MOI.VariableIndex || f isa Float64
            # HiGHS has no VariableIndex or constant objective; the affine
            # one is equivalent.
            f = convert(MOI.ScalarAffineFunction{Float64}, f)
        end
        MOI.set(handle.optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
        Cint(0)
    end
end

# -- Solve and solution retrieval ----------------------------------------------

# Returns Int(MOI.TerminationStatusCode); MOI.OPTIMAL is 1.
Base.@ccallable function jumpy_optimize(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        MOI.optimize!(handle.optimizer)
        Cint(Integer(MOI.get(handle.optimizer, MOI.TerminationStatus())))
    end
end

# Returns Int(MOI.ResultStatusCode) of the primal; MOI.FEASIBLE_POINT is 1.
Base.@ccallable function jumpy_primal_status(model::Ptr{Cvoid})::Cint
    @_catch Cint(-1) begin
        handle = _get(model)
        Cint(Integer(MOI.get(handle.optimizer, MOI.PrimalStatus())))
    end
end

# Writes the primal values of the first min(len, num variables) variables
# into `out`. Returns the number of values written.
Base.@ccallable function jumpy_get_values(
    model::Ptr{Cvoid},
    out::Ptr{Cdouble},
    len::Clonglong,
)::Clonglong
    @_catch Clonglong(-1) begin
        handle = _get(model)
        n = min(len, length(handle.variables))
        values = MOI.get(
            handle.optimizer,
            MOI.VariablePrimal(),
            handle.variables[1:n],
        )
        for k in 1:n
            unsafe_store!(out, values[k], k)
        end
        Clonglong(n)
    end
end

Base.@ccallable function jumpy_objective_value(model::Ptr{Cvoid})::Cdouble
    @_catch Cdouble(NaN) begin
        handle = _get(model)
        Cdouble(MOI.get(handle.optimizer, MOI.ObjectiveValue()))
    end
end

end # module JuMPyHiGHS
