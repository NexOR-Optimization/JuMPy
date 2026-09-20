@testset "native MOI constructor ABI" begin
    ABI = JuMPyHiGHS.JuMPyMOIABI
    MOI = JuMPyHiGHS.MOI

    function native_snf(head::String, args::Vector{UInt64})
        return GC.@preserve head args ABI.jumpy_moi_scalar_nonlinear(
            Cstring(pointer(head)), pointer(args), Clonglong(length(args)),
        )
    end

    @testset "sets and stable kind tags" begin
        sets = (
            MOI.LessThan(2.5), MOI.GreaterThan(2.5), MOI.EqualTo(2.5),
            MOI.ZeroOne(), MOI.Integer(),
        )
        handles = UInt64[]
        for (sense, set) in enumerate(sets)
            h = ABI.jumpy_moi_scalar_set(Cint(sense - 1), 2.5)
            push!(handles, h)
            @test h != 0
            @test ABI.jumpy_moi_kind(h) == sense
            @test ABI.get_object(h) == set
        end
        for (sense, set) in enumerate((MOI.Nonpositives(3), MOI.Nonnegatives(3), MOI.Zeros(3)))
            h = ABI.jumpy_moi_vector_set(Cint(sense - 1), Clonglong(3))
            push!(handles, h)
            @test ABI.jumpy_moi_kind(h) == sense + 5
            @test ABI.get_object(h) == set
        end
        @test allunique(handles)
        for h in handles
            @test ABI.jumpy_moi_release(h) == 0
        end
        empty = ABI.jumpy_moi_vector_set(Cint(2), Clonglong(0))
        @test MOI.dimension(ABI.get_object(empty)) == 0
        @test ABI.jumpy_moi_release(empty) == 0
        @test ABI.jumpy_moi_scalar_set(Cint(-1), 0.0) == 0
        @test ABI.jumpy_moi_scalar_set(Cint(5), 0.0) == 0
        @test ABI.jumpy_moi_vector_set(Cint(3), Clonglong(2)) == 0
        @test ABI.jumpy_moi_vector_set(Cint(0), Clonglong(-1)) == 0
    end

    @testset "native functions and ownership" begin
        x = ABI.jumpy_moi_variable(Clonglong(1))
        c = ABI.jumpy_moi_constant(3.0)
        @test ABI.jumpy_moi_kind(x) == 10
        @test ABI.get_object(x) == MOI.VariableIndex(1)
        @test ABI.jumpy_moi_kind(c) == 9
        @test ABI.get_object(c) === 3.0
        f = native_snf("+", UInt64[x, c])
        g = native_snf("sin", UInt64[f])
        @test f != 0 && g != 0
        @test ABI.jumpy_moi_kind(f) == 11
        borrowed = ABI.get_object(f)
        @test borrowed isa MOI.ScalarNonlinearFunction
        @test borrowed.head == :+
        @test borrowed.args == Any[MOI.VariableIndex(1), 3.0]
        @test ABI.get_object(g).args[1] === borrowed
        @test ABI.jumpy_moi_release(x) == 0
        @test ABI.jumpy_moi_release(c) == 0
        @test ABI.jumpy_moi_release(f) == 0
        GC.gc(true)
        @test ABI.get_object(g).args[1] === borrowed
        @test borrowed.args == Any[MOI.VariableIndex(1), 3.0]
        @test ABI.jumpy_moi_release(g) == 0
        # A JuliaCall borrower keeps the native value alive independently of
        # the registry, and can create a new owned handle without conversion.
        h = ABI.keep(borrowed)
        @test ABI.get_object(h) === borrowed
        @test ABI.jumpy_moi_release(h) == 0
        @test length(borrowed.args) == 2
        @test h > g > f > c > x
        @test_throws ArgumentError ABI.keep(MOI.PositiveSemidefiniteConeTriangle(2))
        @test_throws ArgumentError ABI.keep("not a native MOI node")
        @test ABI.jumpy_moi_variable(Clonglong(0)) == 0
        @test ABI.jumpy_moi_variable(Clonglong(-1)) == 0
    end

    @testset "invalid and stale handles are checked" begin
        @test ABI.jumpy_moi_kind(UInt64(0)) == -1
        @test ABI.jumpy_moi_release(UInt64(0)) == -1
        @test ABI.jumpy_moi_kind(typemax(UInt64)) == -1
        @test_throws ArgumentError ABI.get_object(UInt64(0))
        h = ABI.jumpy_moi_constant(1.0)
        @test ABI.jumpy_moi_release(h) == 0
        @test ABI.jumpy_moi_release(h) == -1
        @test ABI.jumpy_moi_kind(h) == -1
        @test_throws ArgumentError ABI.get_object(h)
        @test native_snf("+", UInt64[h]) == 0
        @test native_snf("+", UInt64[typemax(UInt64)]) == 0
        set = ABI.jumpy_moi_scalar_set(Cint(0), 1.0)
        @test native_snf("+", UInt64[set]) == 0
        @test ABI.jumpy_moi_release(set) == 0
        @test ABI.jumpy_moi_scalar_nonlinear(Cstring(C_NULL), Ptr{UInt64}(C_NULL), Clonglong(0)) == 0
        head = "+"
        GC.@preserve head begin
            ptr = Cstring(pointer(head))
            @test ABI.jumpy_moi_scalar_nonlinear(ptr, Ptr{UInt64}(C_NULL), Clonglong(1)) == 0
            @test ABI.jumpy_moi_scalar_nonlinear(ptr, Ptr{UInt64}(C_NULL), Clonglong(-1)) == 0
            empty = ABI.jumpy_moi_scalar_nonlinear(ptr, Ptr{UInt64}(C_NULL), Clonglong(0))
            @test isempty(ABI.get_object(empty).args)
            @test ABI.jumpy_moi_release(empty) == 0
        end
    end

    @testset "native set consumption by HiGHS" begin
        model = JuMPyHiGHS.jumpy_new_model()
        @test model != C_NULL
        @test JuMPyHiGHS.jumpy_add_variables(model, Clonglong(1)) == 0
        x = JuMPyHiGHS.jumpy_variable(model, Clonglong(0))
        c = JuMPyHiGHS.jumpy_constant(model, 3.0)
        head = "+"
        args = Ptr{Cvoid}[x, c]
        f = GC.@preserve head args JuMPyHiGHS.jumpy_scalar_nonlinear(
            model, Cstring(pointer(head)), pointer(args), Clonglong(2),
        )
        native_set = MOI.LessThan(13.0)
        set = ABI.keep(native_set)
        @test ABI.get_object(set) === native_set
        @test JuMPyHiGHS.jumpy_add_constraint_set(model, f, set) >= 0
        @test ABI.get_object(set).upper == 13.0  # normalization must not change the reusable set
        @test ABI.jumpy_moi_release(set) == 0
        GC.gc(true)
        @test JuMPyHiGHS.jumpy_set_objective_sense(model, Cint(1)) == 0
        @test JuMPyHiGHS.jumpy_set_objective_function(model, x) == 0
        @test JuMPyHiGHS.jumpy_optimize(model) == 1
        @test JuMPyHiGHS.jumpy_objective_value(model) ≈ 10.0
        @test JuMPyHiGHS.jumpy_add_constraint_set(model, x, set) == -1
        vector_set = ABI.jumpy_moi_vector_set(Cint(0), Clonglong(1))
        @test JuMPyHiGHS.jumpy_add_constraint_set(model, x, vector_set) == -1
        @test ABI.jumpy_moi_release(vector_set) == 0
        @test JuMPyHiGHS.jumpy_free_model(model) == 0
    end
end
