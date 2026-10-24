using Test
import JuMPyHiGHS
import GenOpt, HiGHS, JuMP, MathOptInterface

include(joinpath(@__DIR__, "..", "generate_trim_dispatch.jl"))

function _validate_trim_entrypoint(f, types)
    return @test hasmethod(f, Tuple{types...})
end

@testset "finite roots reference existing methods" begin
    path = joinpath(@__DIR__, "..", "trim_roots.jl")
    # entrypoint is a no-op outside image generation. Check its registrations
    # against method lookup without recompiling the image in these unit tests.
    source = replace(
        read(path, String),
        "Base.Experimental.entrypoint" => "_validate_trim_entrypoint",
    )
    include_string(@__MODULE__, source, path)
end

@testset "supplementary dispatch roots" begin
    G = JuMPyTrimDispatch
    mktempdir() do directory
        trace = joinpath(directory, "methods.trace")
        output = joinpath(directory, "roots.jl")
        valid =
            "precompile(Tuple{typeof(GenOpt._check_belongs_to_model), " *
            "JuMPyHiGHS.NativeVariable, JuMPyHiGHS.NativeModel})"
        collection = "precompile(Tuple{typeof(Base.getindex), Array{Any, 1}, Int64})"
        write(
            trace,
            join(
                [
                    valid,
                    collection,
                    valid,
                    "precompile(Tuple{typeof(Main.test_helper), GenOpt.IteratorIndex})",
                    "precompile(Tuple{typeof(JuMPyTrimDispatch._native_dispatch_domain), Type{GenOpt.IteratorIndex}})",
                    "precompile(Tuple{typeof(Test.test_helper), GenOpt.IteratorIndex})",
                    "precompile(Tuple{typeof(Base.Compiler.test_helper), GenOpt.IteratorIndex})",
                    "precompile(Tuple{typeof(MathOptInterface.Bridges.test_helper), Int64})",
                    "precompile(Tuple{typeof(MathOptInterface.Utilities.CachingOptimizer), Int64})",
                    "precompile(Tuple{typeof(JuMPyHiGHS.JuMPyModel.model), HiGHS.Optimizer})",
                    "precompile(Tuple{typeof(Base.sin), Float64})",
                    "unrelated trace output",
                ],
                '\n',
            ),
        )
        @test G.generate(output, [trace]) == 2
        roots = G.dispatch_roots([trace])
        @test length(roots) == 2
        @test roots == sort(roots)
        @test G.dispatch_roots([trace, trace]) == roots
        @test all(startswith("Base.Experimental.entrypoint("), roots)
        @test all(line -> occursin(line, read(output, String)), roots)
        @test Base.include(G, output) === nothing
        @test G.main([output, trace]) === nothing
        @test_throws ErrorException G.main(String[])

        names = filter(
            name -> startswith(string(name), "#append_iterator#_merge_exprs"),
            Base.names(G.GenOpt; all = true),
        )
        wrapper = getfield(G.GenOpt, only(names))
        F = wrapper{IdDict{Base.RefValue{Nothing},Int64},Vector{G.GenOpt.Iterator}}
        observed = Tuple{F,G.GenOpt.Iterator{Tuple{Float64}}}
        write(trace, "precompile(" * repr(observed) * ")")
        @test G.generate(output, [trace]) == length(G.ITERATOR_TUPLES)
        supplemented = G.dispatch_roots([trace])
        @test all(
            T -> "Base.Experimental.entrypoint(" *
            repr(Tuple{F,G.GenOpt.Iterator{T}}) *
            ")" in supplemented,
            G.ITERATOR_TUPLES,
        )

        names = filter(
            name ->
                startswith(string(name), "#_new_values") &&
                getfield(G.GenOpt, name) isa UnionAll,
            Base.names(G.GenOpt; all = true),
        )
        wrapper = getfield(G.GenOpt, only(names))
        F = wrapper{typeof(identity)}
        observed = Tuple{
            typeof(Base.collect_similar),
            Vector{Tuple{Int64}},
            Base.Generator{Vector{Tuple{Int64}},F},
        }
        write(trace, "precompile(" * repr(observed) * ")")
        @test G.generate(output, [trace]) == 1 + 2 * length(G.ITERATOR_TUPLES)
        supplemented = G.dispatch_roots([trace])
        @test all(
            T -> "Base.Experimental.entrypoint(" * repr(Tuple{F,T}) * ")" in supplemented,
            G.ITERATOR_TUPLES,
        )
        @test all(
            T -> "Base.Experimental.entrypoint(" *
            repr(
                Tuple{
                    typeof(Base._collect),
                    Vector{T},
                    Base.Generator{Vector{T},F},
                    Base.EltypeUnknown,
                    Base.HasShape{1},
                },
            ) *
            ")" in supplemented,
            G.ITERATOR_TUPLES,
        )

        destination = Vector{Bool}
        iterator = Base.Generator{Vector{Tuple{Int64}},typeof(G.GenOpt._is_numeric)}
        observed =
            Tuple{typeof(Base.collect_to_with_first!),destination,Bool,iterator,Int64}
        companion = Tuple{typeof(Base.collect_to!),destination,iterator,Int64,Int64}
        write(trace, "precompile(" * repr(observed) * ")")
        @test G.generate(output, [trace]) == 2
        expected = sort([
            "Base.Experimental.entrypoint(" * repr(observed) * ")",
            "Base.Experimental.entrypoint(" * repr(companion) * ")",
        ])
        @test G.dispatch_roots([trace]) == expected
        @test G.dispatch_roots([trace, trace]) == expected
        @test hasmethod(Base.collect_to!, Tuple{destination,iterator,Int64,Int64})

        J = JuMPyHiGHS
        dispatch = typeof(J._dispatch_native)
        callback = typeof(identity)
        constraints = Union{J.NativeScalar,J.NativeTemplate}
        entrypoint(T) = "Base.Experimental.entrypoint(" * repr(T) * ")"
        for domains in (
            (J.NativeScalar,),
            (J.NativeNode,),
            (J.NativeNode, J.NativeNode),
            (constraints, J.NativeSet),
        )
            domain_types = map(Core.Typeof, domains)
            values = map(T -> first(Base.uniontypes(T)), domains)
            observed = Tuple{dispatch,callback,values...,domain_types...}
            write(trace, "precompile(" * repr(observed) * ")")
            # Only types are retained; identity is never invoked as a callback.
            expected = sort!(
                vec([
                    entrypoint(Tuple{dispatch,callback,types...,domain_types...}) for
                    types in Iterators.product(map(Base.uniontypes, domains)...)
                ]),
            )
            @test G.generate(output, [trace]) == length(expected)
            @test G.dispatch_roots([trace]) == expected
            @test G.dispatch_roots([trace, trace]) == expected
        end
        # The observed constraint pair above does not include this template.
        unobserved = Tuple{
            dispatch,
            callback,
            GenOpt.ExprTemplate{J.NativeVariable,J.NativeVariable},
            MathOptInterface.LessThan{Float64},
            Core.Typeof(constraints),
            Core.Typeof(J.NativeSet),
        }
        @test entrypoint(unobserved) in G.dispatch_roots([trace])
        retained = String[]
        seen = Set{Any}()
        G._supplement_native_dispatch_roots!(retained, unobserved, seen)
        count = length(retained)
        @test length(seen) == 1
        G._supplement_native_dispatch_roots!(retained, unobserved, seen)
        @test length(retained) == count
        @test G._native_dispatch_domain(Type{J.NativeScalar}) === J.NativeScalar
        @test G._native_dispatch_domain(Type{J.NativeSet}) === J.NativeSet
        @test G._native_dispatch_domain(Core.Typeof(J.NativeScalar)) === J.NativeScalar
        @test G._native_dispatch_domain(Core.Typeof(J.NativeSet)) === J.NativeSet
        @test G._native_dispatch_domain(Int64) === nothing
        @test G._native_dispatch_domain(Type{Any}) === nothing
        @test G._native_dispatch_domain(Type{Real}) === nothing

        skipped = String[]
        for signature in (
            Tuple,
            Tuple{},
            Tuple{dispatch,callback,Int64},
            Tuple{typeof(identity),callback,Int64,Type{J.NativeScalar}},
            Tuple{dispatch,callback,Int64,Type{Any}},
            Tuple{dispatch,callback,Int64,Int64},
        )
            @test G._supplement_native_dispatch_roots!(skipped, signature) === nothing
        end
        @test isempty(skipped)

        roots = String[]
        @test G._supplement_collect_roots!(roots, Tuple) === nothing
        @test G._supplement_collect_roots!(
            roots,
            Tuple{typeof(identity),Int64,Int64,Int64,Int64},
        ) === nothing
        @test isempty(roots)

        write(trace, "precompile(Tuple{typeof(GenOpt.unknown_dispatch_target), Int64})")
        @test_throws ErrorException G.generate(output, [trace])
        write(trace, "precompile(Tuple{typeof(GenOpt._expand),)")
        @test_throws ErrorException G.generate(output, [trace])
        write(trace, "unrelated trace output")
        @test_throws ErrorException G.generate(output, [trace])
    end
end
