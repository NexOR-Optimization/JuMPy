using JuMPyHiGHS

# -- Ordinary-Julia tracing workload --------------------------------------------

# Run this in ordinary Julia to trace model-building and solution paths.
# Image generation does not initialize the solver JLL, so this is deliberately
# not included by juliac_entry.jl.
let J = JuMPyHiGHS
    function apply(m, head::String, args::Ptr{Cvoid}...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_apply(
            m, Cstring(pointer(head)), pointer(argv), Clonglong(length(argv)),
        )
    end
    function add(m, func, set)
        result = J.jumpy_add_constraint(m, func, set)
        J.jumpy_free_set(set)
        return result
    end
    m = J.jumpy_new_model()
    @assert J.jumpy_set_time_limit_sec(m, 10.0) == 0
    @assert J.jumpy_set_time_limit_sec(m, NaN) == 0
    variables = J.jumpy_variables(m, Clonglong(3))
    x = apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(1)))
    # bounds and integrality: every set
    add(m, x, J.jumpy_greater_than(0.0))
    add(m, x, J.jumpy_less_than(10.0))
    add(m, apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(2))), J.jumpy_zero_one())
    add(m, apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(3))), J.jumpy_integer())
    # affine expressions with every operator Python emits
    one = J.jumpy_constant(m, 1.0)
    plus = apply(m, "+", x, one)
    add(m, plus, J.jumpy_less_than(5.0))
    add(m, plus, J.jumpy_greater_than(0.0))
    add(m, plus, J.jumpy_equal_to(3.0))
    scaled = apply(m, "*", J.jumpy_constant(m, 2.0), apply(m, "-", x, one))
    add(m, apply(m, "/", scaled, J.jumpy_constant(m, 4.0)), J.jumpy_less_than(8.0))
    # groups: iterator, variable block, data array, 1-D and 2-D
    values = Clonglong[0, 1, 2]
    i = GC.@preserve values J.jumpy_integer_iterator(m, pointer(values), Clonglong(3))
    block = J.jumpy_variables(m, Clonglong(3))
    data = Cdouble[1.0, 2.0, 3.0]
    d = GC.@preserve data J.jumpy_float_array(m, pointer(data), Clonglong(3))
    i1 = apply(m, "+", i, J.jumpy_integer_constant(m, Clonglong(1)))
    template = apply(m, "-", apply(m, "getindex", block, i1), apply(m, "getindex", d, i1))
    set = J.jumpy_greater_than(0.0)
    J.jumpy_add_constraint(m, template, set)
    J.jumpy_free_set(set)
    set = J.jumpy_greater_than(1.0)
    J.jumpy_add_constraint(m, apply(m, "getindex", block, i1), set)
    J.jumpy_free_set(set)
    pair = Cdouble[0.0, 1.0]
    j = GC.@preserve pair J.jumpy_iterator(m, pointer(pair), Clonglong(2))
    template2 = apply(m, "-", apply(m, "+", apply(m, "getindex", block, i1), j), one)
    set = J.jumpy_less_than(5.0)
    J.jumpy_add_constraint(m, template2, set)
    J.jumpy_free_set(set)
    # objective: both senses, function and bare-variable forms
    J.jumpy_set_objective_sense(m, Cint(0))
    J.jumpy_set_objective_function(m, plus)
    J.jumpy_set_objective_sense(m, Cint(1))
    J.jumpy_set_objective_function(m, x)
    J.jumpy_optimize(m)
    J.jumpy_value(m, x)
    J.jumpy_value(m, plus)
    # Nonlinear expressions can be evaluated at a linear model's solution,
    # even though HiGHS does not accept nonlinear constraints/objectives.
    for head in ("sin", "cos", "exp", "log", "sqrt", "abs")
        nonlinear = apply(m, head, x)
        nested = apply(m, "+", nonlinear, apply(m, head, plus))
        @assert isfinite(J.jumpy_value(m, nonlinear))
        @assert isfinite(J.jumpy_value(m, nested))
    end
    # every remaining operator Python can emit (nonlinear ones end in the
    # unsupported-constraint error path, which is also worth exercising)
    add(m, apply(m, "-", x), J.jumpy_less_than(1.0))
    add(m, apply(m, "^", x, J.jumpy_constant(m, 2.0)), J.jumpy_less_than(9.0))
    for f in ("sin", "cos", "exp", "log", "sqrt", "abs")
        add(m, apply(m, f, x), J.jumpy_less_than(1.0))
        generated = apply(m, f, apply(m, "getindex", block, i1))
        @assert add(m, generated, J.jumpy_less_than(1.0)) == -1
    end
    # Errors remain contained within the C entry points.
    J.jumpy_optimize(C_NULL)
    J.jumpy_free_model(m)

    J.jumpy_free_set(UInt64(0))
    J.jumpy_add_constraint(C_NULL, C_NULL, UInt64(0))
end
