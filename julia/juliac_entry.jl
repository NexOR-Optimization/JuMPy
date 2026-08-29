# JuliaC entry script: build with
#
#     juliac --output-lib build/libjumpy_highs --project . juliac_entry.jl \
#         --compile-ccallable --jl-option handle-signals=no \
#         --experimental --trim=unsafe-warn --bundle build
#
# The `Base.Experimental.entrypoint` declarations below must execute in the
# image-build session (module top-level code runs at package-precompile
# time, and entrypoint registrations do not survive the cache).

using JuMPyHiGHS
import GenOpt
import HiGHS
import MathOptInterface
import MathOptInterface as MOI

# Dynamic dispatch inside MOI/GenOpt (from values inference only knows as
# Any — e.g. ScalarNonlinearFunction args) can only succeed in a trimmed
# image if the target specialization was compiled. Declare every
# combination the ABI can produce as an entry point.
for F in (
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarNonlinearFunction,
    ),
    S in (
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.ZeroOne,
        MOI.Integer,
    )

    Base.Experimental.entrypoint(
        MOI.Utilities.normalize_and_add_constraint,
        (JuMPyHiGHS.Optimizer, F, S),
    )
end
for F in (MOI.ScalarAffineFunction{Float64}, MOI.ScalarNonlinearFunction)
    Base.Experimental.entrypoint(
        MOI.set,
        (JuMPyHiGHS.Optimizer, MOI.ObjectiveFunction{F}, F),
    )
end
Base.Experimental.entrypoint(
    MOI.Nonlinear.SymbolicAD.simplify,
    (MOI.ScalarNonlinearFunction,),
)
# ScalarNonlinearFunction args are Vector{Any}: code that maps over them
# (copy in simplify, substitution in GenOpt._expand) dispatches dynamically
# per element, so each leaf specialization needs to be compiled explicitly.
for T in (
        Float64,
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarNonlinearFunction,
        GenOpt.IteratorRef,
        GenOpt.IteratorIndex,
        GenOpt.ContiguousArrayOfVariables{1},
        Vector{Float64},
    )
    Base.Experimental.entrypoint(Base.copy, (T,))
    Base.Experimental.entrypoint(GenOpt._expand, (T, Vector{Float64}))
end

# Plain Base operations on Vector{Any} (the args of every
# ScalarNonlinearFunction): reached through dynamic dispatch all over MOI,
# and invisible to the trace when already compiled in package caches.
for (f, args) in (
        (Base.length, (Vector{Any},)),
        (Base.isempty, (Vector{Any},)),
        (Base.getindex, (Vector{Any}, Int64)),
        (Base.setindex!, (Vector{Any}, Float64, Int64)),
        (Base.iterate, (Vector{Any},)),
        (Base.iterate, (Vector{Any}, Int64)),
        (Base.eachindex, (Vector{Any},)),
        (Base.similar, (Vector{Any},)),
        (Base.copy, (Vector{Any},)),
        (Base.Iterators.only, (Tuple{MOI.ScalarNonlinearFunction},)),
    )
    Base.Experimental.entrypoint(f, args)
end

include("trim_dispatch.jl")

include("workload.jl")
