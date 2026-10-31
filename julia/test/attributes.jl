using Test
import JuMP
import JuMPyHiGHS
import MathOptInterface as MOI

@testset "raw optimizer attributes through the C entry points" begin
    J = JuMPyHiGHS
    m = J.jumpy_new_model()
    optimizer = JuMP.backend(J._get(m).model)
    for (name, value, expected) in (
        ("presolve", "off", "off"),
        ("mip_detect_symmetry", false, false),
        ("random_seed", 3, 3),
        ("time_limit", 5.0, 5.0),
    )
        status = GC.@preserve name begin
            key = Cstring(pointer(name))
            if value isa String
                GC.@preserve value J.jumpy_set_string_attribute(m, key, Cstring(pointer(value)))
            elseif value isa Bool
                J.jumpy_set_bool_attribute(m, key, Cint(value))
            elseif value isa Int
                J.jumpy_set_int_attribute(m, key, Clonglong(value))
            else
                J.jumpy_set_float_attribute(m, key, Cdouble(value))
            end
        end
        @test status == 0
        @test MOI.get(optimizer, MOI.RawOptimizerAttribute(name)) == expected
    end
    name = "not_an_option"
    redirect_stderr(devnull) do
        @test GC.@preserve name J.jumpy_set_int_attribute(
            m,
            Cstring(pointer(name)),
            Clonglong(1),
        ) == -1
    end
    @test J.jumpy_free_model(m) == 0
end
