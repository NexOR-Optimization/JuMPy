# -- Ordinary-Julia tracing workload --------------------------------------------

# Run this in ordinary Julia to trace model-building and solution paths.
# Image generation does not initialize the solver JLL, so this is deliberately
# not included by juliac_entry.jl. With JUMPY_BACKEND=nexor, it traces the
# NexOR client instead, answering the solve with a mock transport.
if get(ENV, "JUMPY_BACKEND", "") == "nexor"
    include("nexor_entry.jl")
else
    using JuMPyHiGHS
end

let J = isdefined(Main, :JuMPyNexOR) ? JuMPyNexOR : JuMPyHiGHS
    nexor = isdefined(Main, :JuMPyNexOR)
    function apply(m, head::String, args::Ptr{Cvoid}...)
        argv = collect(Ptr{Cvoid}, args)
        return GC.@preserve head argv J.jumpy_apply(
            m,
            Cstring(pointer(head)),
            pointer(argv),
            Clonglong(length(argv)),
        )
    end
    function add(m, func, set)
        result = J.jumpy_add_constraint(m, func, set)
        J.jumpy_free_set(set)
        return result
    end
    function set_attribute(m, name::String, value)
        GC.@preserve name value begin
            key = Cstring(pointer(name))
            if value isa String
                J.jumpy_set_string_attribute(m, key, Cstring(pointer(value)))
            elseif value isa Bool
                J.jumpy_set_bool_attribute(m, key, Cint(value))
            elseif value isa Int
                J.jumpy_set_int_attribute(m, key, Clonglong(value))
            else
                J.jumpy_set_float_attribute(m, key, Cdouble(value))
            end
        end
    end
    function optimize(m)
        nexor || return J.jumpy_optimize(m)
        model = J._get(m).model
        client = J.JuMP.backend(model)
        primal = ones(J.MOI.get(client, J.MOI.NumberOfVariables()))
        function request(_, method, path; body = "")
            path == "/problems" && return Dict("problem_id" => "w", "status" => "queued")
            startswith(path, "/problems/w/wait") &&
                return Dict("problem_id" => "w", "status" => "finished")
            attributes = Dict(
                "termination_status" => "OPTIMAL",
                "primal_status" => "FEASIBLE_POINT",
                "dual_status" => "NO_SOLUTION",
                "result_count" => 1,
                "objective_value" => 1.0,
                "raw_status" => "workload",
            )
            return Dict("solution" => Dict("attributes" => attributes, "primal" => primal))
        end
        NexOR._optimize!(client, request)
        # The mock transport bypasses the JuMP optimize wrapper.
        model.is_model_dirty = false
        return J.jumpy_termination_status(m)
    end
    m = J.jumpy_new_model()
    @assert set_attribute(m, nexor ? "solver" : "presolve", nexor ? "HiGHS" : "on") == 0
    @assert set_attribute(m, "mip_detect_symmetry", true) == 0
    @assert set_attribute(m, "random_seed", 1) == 0
    @assert set_attribute(m, "mip_rel_gap", 1e-4) == 0
    @assert J.jumpy_set_time_limit_sec(m, 10.0) == 0
    @assert J.jumpy_set_time_limit_sec(m, NaN) == 0
    variables = J.jumpy_variables(m, Clonglong(3))
    x = apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(1)))
    # bounds and integrality: every set
    add(m, x, J.jumpy_greater_than(0.0))
    add(m, x, J.jumpy_less_than(10.0))
    add(
        m,
        apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(2))),
        J.jumpy_zero_one(),
    )
    add(
        m,
        apply(m, "getindex", variables, J.jumpy_integer_constant(m, Clonglong(3))),
        J.jumpy_integer(),
    )
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
    optimize(m)
    J.jumpy_has_values(m)
    J.jumpy_objective_value(m)
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
        # HiGHS rejects nonlinear constraints; NexOR sends them to the server.
        @assert (add(m, generated, J.jumpy_less_than(1.0)) == C_NULL) != nexor
    end
    # Lazy sums and quadratic groups (HiGHS rejects the latter).
    add(m, apply(m, "lazy_sum", apply(m, "getindex", block, i1)), J.jumpy_less_than(9.0))
    squares =
        apply(m, "*", apply(m, "getindex", block, i1), apply(m, "getindex", block, i1))
    add(m, squares, J.jumpy_less_than(4.0))
    # Errors remain contained within the C entry points.
    J.jumpy_optimize(C_NULL)
    J.jumpy_free_model(m)

    J.jumpy_free_set(UInt64(0))
    J.jumpy_add_constraint(C_NULL, C_NULL, UInt64(0))
end
