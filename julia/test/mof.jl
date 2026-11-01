using Test
import JSON
import JuMP
import JuMPyHiGHS
import MathOptInterface as MOI
if !isdefined(@__MODULE__, :JuMPyNexOR)
    include("../src/JuMPyNexOR.jl")
end

@testset "local MOF export and variable counts" begin
    for J in (JuMPyHiGHS, JuMPyNexOR)
        m = J.jumpy_new_model()
        try
            @test J.jumpy_num_variables(m) == 0
            block = J.jumpy_variables(m, Clonglong(4))
            x, y, z, w = J._unbox(block)
            model = J._get(m).model
            J.JuMPyDirect.add_constraint(model, x, MOI.GreaterThan(1.0))
            J.JuMPyDirect.add_constraint(model, x, MOI.LessThan(4.0))
            J.JuMPyDirect.add_constraint(model, y, MOI.ZeroOne())
            J.JuMPyDirect.add_constraint(model, z, MOI.Integer())
            J.JuMPyDirect.add_constraint(model, w, MOI.EqualTo(2.0))
            J.JuMPyDirect.add_constraint(model, x + 2y, MOI.LessThan(4.0))
            J.JuMPyDirect.add_constraint(model, x + 2y, MOI.GreaterThan(1.0))
            J.JuMPyDirect.add_constraint(model, x + 2y, MOI.EqualTo(2.0))
            if J === JuMPyNexOR
                J.JuMPyDirect.add_constraint(model, x^2 + y, MOI.LessThan(10.0))
                for set in (MOI.LessThan(1.0), MOI.GreaterThan(-1.0), MOI.EqualTo(0.0))
                    J.JuMPyDirect.add_constraint(model, sin(x), set)
                end
            end
            J.JuMPyDirect.set_objective_sense(model, MOI.MIN_SENSE)
            J.JuMPyDirect.set_objective_function(model, x^2 + 3y)
            @test J.jumpy_num_variables(m) == 4
            mktempdir() do directory
                path = joinpath(directory, "model.mof.json")
                @test GC.@preserve path J.jumpy_write_mof(m, Cstring(pointer(path))) == 0
                snapshot = MOI.FileFormats.MOF.Model()
                MOI.read_from_file(snapshot, path)
                @test MOI.get(snapshot, MOI.NumberOfVariables()) == 4
                @test length(JSON.parsefile(path)["constraints"]) ==
                      (J === JuMPyNexOR ? 12 : 8)
                @test MOI.get(snapshot, MOI.ObjectiveSense()) == MOI.MIN_SENSE
                @test MOI.get(snapshot, MOI.ObjectiveFunctionType()) ==
                      MOI.ScalarQuadraticFunction{Float64}
                # Writing never optimizes, including with an unset remote solver.
                @test JuMP.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
                invalid = joinpath(directory, "missing", "model.mof.json")
                redirect_stderr(devnull) do
                    @test GC.@preserve invalid J.jumpy_write_mof(
                        m,
                        Cstring(pointer(invalid)),
                    ) == -1
                    @test J.jumpy_num_variables(C_NULL) == -1
                    @test GC.@preserve path J.jumpy_write_mof(
                        C_NULL,
                        Cstring(pointer(path)),
                    ) == -1
                end
            end
        finally
            J.jumpy_free_model(m)
        end
    end
    # The shared helper also serves cached JuliaCall models without a solver.
    J = JuMPyHiGHS.JuMPyModel
    model = J.model()
    x = only(J.add_variables(model, 1))
    J.add_constraint(model, x, MOI.GreaterThan(2.0))
    J.set_objective_sense(model, MOI.MIN_SENSE)
    J.set_objective_function(model, x)
    mktempdir() do directory
        path = joinpath(directory, "cached.mof.json")
        @test J.write_mof(model, path) === nothing
        @test J.num_variables(model) == 1
        @test length(JSON.parsefile(path)["variables"]) == 1
    end
end

function _mof_apply(J, m, op::String, args...)
    argv = Ptr{Cvoid}[args...]
    return GC.@preserve op argv J.jumpy_apply(
        m,
        Cstring(pointer(op)),
        pointer(argv),
        Clonglong(length(argv)),
    )
end

@testset "NexOR grouped sums with mixed numeric coefficients" begin
    J = JuMPyNexOR
    m = J.jumpy_new_model()
    try
        block = J.jumpy_variables(m, Clonglong(9))
        values = Clonglong[1, 2, 3]
        i = GC.@preserve values J.jumpy_integer_iterator(m, pointer(values), Clonglong(3))
        constant(value) = J.jumpy_integer_constant(m, Clonglong(value))
        expression = _mof_apply(J, m, "getindex", block, i)
        for offset in (3, 6)
            column = _mof_apply(J, m, "+", i, constant(offset))
            term = _mof_apply(J, m, "getindex", block, column)
            expression = _mof_apply(J, m, "+", expression, term)
        end
        expression = _mof_apply(J, m, "-", expression, constant(1))
        set = J.jumpy_equal_to(0.0)
        try
            @test J.jumpy_add_constraint(m, expression, set) != C_NULL
        finally
            J.jumpy_free_set(set)
        end
        mktempdir() do directory
            path = joinpath(directory, "groups.mof.json")
            @test GC.@preserve path J.jumpy_write_mof(m, Cstring(pointer(path))) == 0
            data = JSON.parsefile(path)
            @test length(data["variables"]) == 9
            @test length(data["constraints"]) == 3
            @test all(row -> length(row["function"]["terms"]) == 3, data["constraints"])
        end
    finally
        J.jumpy_free_model(m)
    end
end
