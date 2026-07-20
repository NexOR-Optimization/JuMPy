# Build-time workload, executed both by juliac_entry.jl (so the exact
# runtime specializations exist in the build session) and standalone under
# --trace-dispatch to generate trim_dispatch.jl:
#
#     julia --project=. --trace-dispatch=/tmp/dispatch.jl workload.jl
#
# It must NOT include trim_dispatch.jl: pre-registered entrypoints are
# compiled upfront and would hide their dispatches from the trace.

using JuMPyHiGHS

# -- Build-time workload --------------------------------------------------------

# Exercise every ABI path once in the build session: everything executed
# here is compiled with exactly the specializations the runtime will need,
# which covers dynamic-dispatch targets no entrypoint declaration names.
let J = JuMPyHiGHS
    function snf(m, head::String, args::Ptr{Cvoid}...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_scalar_nonlinear(
            m, Cstring(pointer(head)), pointer(argv), Clonglong(length(argv)),
        )
    end
    m = J.jumpy_new_model()
    J.jumpy_add_variables(m, Clonglong(6))
    x = J.jumpy_variable(m, Clonglong(0))
    # bounds and integrality: every set
    J.jumpy_add_constraint(m, x, Cint(1), 0.0)
    J.jumpy_add_constraint(m, x, Cint(0), 10.0)
    J.jumpy_add_constraint(m, J.jumpy_variable(m, Clonglong(1)), Cint(3), 0.0)
    J.jumpy_add_constraint(m, J.jumpy_variable(m, Clonglong(2)), Cint(4), 0.0)
    # affine expressions with every operator Python emits, every sense
    one = J.jumpy_constant(m, 1.0)
    plus = snf(m, "+", x, one)
    J.jumpy_add_constraint(m, plus, Cint(0), 5.0)
    J.jumpy_add_constraint(m, plus, Cint(1), 0.0)
    J.jumpy_add_constraint(m, plus, Cint(2), 3.0)
    scaled = snf(m, "*", J.jumpy_constant(m, 2.0), snf(m, "-", x, one))
    J.jumpy_add_constraint(m, snf(m, "/", scaled, J.jumpy_constant(m, 4.0)), Cint(0), 8.0)
    # groups: iterator, variable block, data array, 1-D and 2-D
    values = Cdouble[0.0, 1.0, 2.0]
    i = GC.@preserve values J.jumpy_iterator(m, pointer(values), Clonglong(3))
    block = J.jumpy_contiguous_variables(m, Clonglong(3), Clonglong(3))
    data = Cdouble[1.0, 2.0, 3.0]
    d = GC.@preserve data J.jumpy_float_array(m, pointer(data), Clonglong(3))
    i1 = snf(m, "+", i, one)
    template = snf(m, "-", snf(m, "getindex", block, i1), snf(m, "getindex", d, i1))
    J.jumpy_add_group_constraint(m, template, Cint(1))
    pair = Cdouble[0.0, 1.0]
    j = GC.@preserve pair J.jumpy_iterator(m, pointer(pair), Clonglong(2))
    template2 = snf(m, "-", snf(m, "+", snf(m, "getindex", block, i1), j), one)
    J.jumpy_add_group_constraint(m, template2, Cint(0))
    # objective: both senses, function and bare-variable forms
    J.jumpy_set_objective_sense(m, Cint(0))
    J.jumpy_set_objective_function(m, plus)
    J.jumpy_set_objective_sense(m, Cint(1))
    J.jumpy_set_objective_function(m, x)
    J.jumpy_optimize(m)
    J.jumpy_primal_status(m)
    out = zeros(Cdouble, 6)
    GC.@preserve out J.jumpy_get_values(m, pointer(out), Clonglong(6))
    J.jumpy_objective_value(m)
    # every remaining operator Python can emit (nonlinear ones end in the
    # unsupported-constraint error path, which is also worth exercising)
    J.jumpy_add_constraint(m, snf(m, "-", x), Cint(0), 1.0)
    J.jumpy_add_constraint(m, snf(m, "^", x, J.jumpy_constant(m, 2.0)), Cint(0), 9.0)
    for f in ("sin", "cos", "exp", "log", "sqrt", "abs")
        J.jumpy_add_constraint(m, snf(m, f, x), Cint(0), 1.0)
    end
    # error paths (print through jl_static_show)
    J.jumpy_optimize(C_NULL)
    J.jumpy_free_model(m)
end
