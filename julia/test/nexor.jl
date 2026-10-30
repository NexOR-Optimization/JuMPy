# Run on its own with `julia --project=test test/nexor.jl` from julia/.
include("../nexor_entry.jl")
module NexORTests

using Test
import JSON
import MathOptInterface as MOI
import NexOR
import ..JuMPyNexOR
const Backend = JuMPyNexOR

optimizer(m) = Backend.JuMP.backend(Backend._get(m).model)
constant(m, value::Integer) = Backend.jumpy_integer_constant(m, Clonglong(value))
constant(m, value) = Backend.jumpy_constant(m, Cdouble(value))

function apply(m, head::String, args...)
    argv = collect(Ptr{Cvoid}, args)
    return GC.@preserve head argv Backend.jumpy_apply(
        m,
        Cstring(pointer(head)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

variable(m, block, i) = apply(m, "getindex", block, constant(m, i))

function add_constraint(m, func, set)
    id = Backend._store_set(set)
    try
        return Backend.jumpy_add_constraint(m, func, id)
    finally
        Backend.jumpy_free_set(id)
    end
end

@testset "NexOR configuration" begin
    withenv(
        "NEXOR_SERVER_URL" => "https://example.test",
        "NEXOR_API_KEY" => "test-key",
        "NEXOR_SOLVER" => "Ipopt",
    ) do
        m = Backend.jumpy_new_model()
        @test m != C_NULL
        try
            client = optimizer(m)
            @test client.server_url == "https://example.test"
            @test client.api_key == "test-key"
            @test client.solver.optimizer == "Ipopt"
            @test MOI.get(client, MOI.Silent())
            @test optimizer(m) isa NexOR.Optimizer
            @test MOI.supports_incremental_interface(client)
            @test MOI.get(client, MOI.TimeLimitSec()) === nothing
            @test Backend.jumpy_set_time_limit_sec(m, 30.0) == 0
            @test MOI.get(client, MOI.TimeLimitSec()) == 30.0
            @test Backend.jumpy_set_time_limit_sec(m, NaN) == 0
            @test MOI.get(client, MOI.TimeLimitSec()) === nothing
        finally
            @test Backend.jumpy_free_model(m) == 0
        end
    end
    withenv("NEXOR_SOLVER" => nothing) do
        @test Backend._new_optimizer().solver.optimizer == "HiGHS"
    end
end

@testset "NexOR native handles and MOF round trip" begin
    m = Backend.jumpy_new_model()
    @test m != C_NULL
    try
        block = Backend.jumpy_variables(m, Clonglong(3))
        @test block != C_NULL
        variables = Backend._unbox(block)
        generated = Backend.GenOpt.to_generator(variables)
        @test collect(generated) == variables
        @test Backend.JuMP.moi_function(generated) == Backend.JuMP.index.(variables)
        other = Backend.jumpy_new_model()
        try
            @test_throws Backend.JuMP.VariableNotOwned Backend.JuMP.moi_function(
                Backend._get(other).model,
                generated,
            )
        finally
            @test Backend.jumpy_free_model(other) == 0
        end
        x, y, z = (variable(m, block, i) for i = 1:3)
        @test add_constraint(m, x, MOI.ZeroOne()) != C_NULL
        @test add_constraint(m, y, MOI.Integer()) != C_NULL
        values = Clonglong[0, 1, 2]
        demand = Cdouble[1, 2, 3]
        i = GC.@preserve values Backend.jumpy_integer_iterator(
            m,
            pointer(values),
            Clonglong(3),
        )
        d = GC.@preserve demand Backend.jumpy_float_array(m, pointer(demand), Clonglong(3))
        # Python shifts symbolic indices before indexing variables and data.
        index = apply(m, "+", i, constant(m, 1))
        data = apply(m, "getindex", d, index)
        @test data != C_NULL
        @test Backend._unbox(data) isa Backend.GenOpt.IteratorValues{Float64}
        template = apply(m, "-", apply(m, "getindex", block, index), data)
        @test add_constraint(m, template, MOI.GreaterThan(0.0)) != C_NULL
        objective = apply(m, "+", x, apply(m, "+", y, z))
        @test Backend.jumpy_set_objective_sense(m, Cint(0)) == 0
        @test Backend.jumpy_set_objective_function(m, objective) == 0
        client = optimizer(m)
        envelope = JSON.parse(NexOR._submission(client))
        @test envelope["problem_type"] == "opaque/1"
        @test envelope["solver"] == "highs-mof"
        @test envelope["options"]["solver_options"]["highs-mof"]["silent"]
        problem = envelope["problem"]
        @test length(problem["variables"]) == 3
        @test length(problem["constraints"]) == 5
        restored = MOI.FileFormats.MOF.Model()
        read!(IOBuffer(JSON.json(problem)), restored)
        @test MOI.get(restored, MOI.NumberOfVariables()) == 3
        @test MOI.get(restored, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.ZeroOne}()) ==
              1
        @test MOI.get(restored, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Integer}()) ==
              1
        @test MOI.get(
            restored,
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64},
            }(),
        ) == 3
        rows = MOI.get(
            restored,
            MOI.ListOfConstraintIndices{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64},
            }(),
        )
        for (i, row) in enumerate(rows)
            func = MOI.get(restored, MOI.ConstraintFunction(), row)
            @test only(func.terms).variable == MOI.VariableIndex(i)
            @test only(func.terms).coefficient == 1
            @test func.constant == 0
            @test MOI.get(restored, MOI.ConstraintSet(), row).lower == i
        end
        requests = String[]
        function request(model, method, path; body = "")
            push!(requests, method * " " * path)
            if path == "/problems"
                @test JSON.parse(body)["problem"] == problem
                return Dict("problem_id" => "test", "status" => "queued")
            elseif startswith(path, "/problems/test/wait")
                return Dict("problem_id" => "test", "status" => "finished")
            end
            @test path == "/problems/test/result?inline=1"
            return Dict(
                "solution" => Dict(
                    "attributes" => Dict(
                        "termination_status" => "OPTIMAL",
                        "primal_status" => "FEASIBLE_POINT",
                        "dual_status" => "NO_SOLUTION",
                        "result_count" => 1,
                        "objective_value" => 6.0,
                        "raw_status" => "test solution",
                    ),
                    "primal" => [1.0, 2.0, 3.0],
                ),
            )
        end
        NexOR._optimize!(client, request)
        # The mock transport bypasses the JuMP optimize wrapper.
        Backend._get(m).model.is_model_dirty = false
        @test length(requests) == 3
        @test Backend.jumpy_termination_status(m) == 1
        @test Backend.jumpy_has_values(m) == 1
        @test Backend.jumpy_has_duals(m) == 0
        @test Backend.jumpy_objective_value(m) == 6.0
        @test [Backend.jumpy_value(m, v) for v in (x, y, z)] == [1.0, 2.0, 3.0]
        # Editing the model invalidates the previous result.
        @test add_constraint(m, apply(m, "sin", z), MOI.LessThan(1.0)) != C_NULL
        @test MOI.get(client, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(client, MOI.ResultCount()) == 0
        @test Backend.jumpy_has_values(m) == 0
        @test client.problem_id === nothing
        problem = NexOR._problem(client)
        @test problem["has_scalar_nonlinear"]
        @test any(
            c -> c["function"]["type"] == "ScalarNonlinearFunction",
            problem["constraints"],
        )
        redirect_stderr(devnull) do
            @test Backend.jumpy_optimize(C_NULL) == -1
        end
    finally
        @test Backend.jumpy_free_model(m) == 0
    end
    @test isempty(Backend.KEEP_ALIVE)
end

@testset "NexOR objective shapes" begin
    for kind in (:constant, :zero, :variable, :quadratic, :nonlinear, :lazy_sum)
        m = Backend.jumpy_new_model()
        try
            block = Backend.jumpy_variables(m, Clonglong(1))
            x = variable(m, block, 1)
            func, expected = if kind == :constant
                constant(m, 2.0), "ScalarAffineFunction"
            elseif kind == :zero
                apply(m, "-", x, x), "ScalarAffineFunction"
            elseif kind == :variable
                x, "ScalarAffineFunction"
            elseif kind == :quadratic
                apply(m, "*", x, x), "ScalarQuadraticFunction"
            elseif kind == :nonlinear
                apply(m, "sin", x), "ScalarNonlinearFunction"
            else
                indices = Clonglong[1]
                i = GC.@preserve indices Backend.jumpy_integer_iterator(
                    m,
                    pointer(indices),
                    Clonglong(1),
                )
                apply(m, "lazy_sum", apply(m, "getindex", block, i)), "ScalarAffineFunction"
            end
            @test Backend.jumpy_set_objective_sense(m, Cint(1)) == 0
            @test Backend.jumpy_set_objective_function(m, func) == 0
            objective = NexOR._problem(optimizer(m))["objective"]
            @test objective["sense"] == "max"
            @test objective["function"]["type"] == expected
        finally
            @test Backend.jumpy_free_model(m) == 0
        end
    end
end

@testset "Cartesian products and empty native groups" begin
    m = Backend.jumpy_new_model()
    try
        block = Backend.jumpy_variables(m, Clonglong(4))
        indices = Clonglong[0, 1]
        i = GC.@preserve indices Backend.jumpy_integer_iterator(
            m,
            pointer(indices),
            Clonglong(2),
        )
        j = GC.@preserve indices Backend.jumpy_integer_iterator(
            m,
            pointer(indices),
            Clonglong(2),
        )
        offset = apply(
            m,
            "+",
            apply(m, "+", apply(m, "*", constant(m, 2), i), j),
            constant(m, 1),
        )
        template = apply(m, "getindex", block, offset)
        @test add_constraint(m, template, MOI.GreaterThan(0.0)) != C_NULL
        @test MOI.get(
            optimizer(m),
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64},
            }(),
        ) == 4
        empty_values = Clonglong[]
        empty_i = GC.@preserve empty_values Backend.jumpy_integer_iterator(
            m,
            pointer(empty_values),
            Clonglong(0),
        )
        @test add_constraint(
            m,
            apply(m, "getindex", block, empty_i),
            MOI.GreaterThan(0.0),
        ) != C_NULL
        @test MOI.get(
            optimizer(m),
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64},
            }(),
        ) == 4
    finally
        @test Backend.jumpy_free_model(m) == 0
    end
end

end # module NexORTests
