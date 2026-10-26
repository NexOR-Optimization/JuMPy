# Generate supplementary compilation roots from ordinary Julia runs, using
# the exact Julia version and dependency environment of the image build:
#
#   julia --project=test --trace-dispatch=tests.trace test/runtests.jl
#   julia --project=test --trace-dispatch=workload.trace workload.jl
#   julia --project=test generate_trim_dispatch.jl trim_dispatch.jl \
#       tests.trace workload.trace
#
# Do not execute solver workloads while generating a nonincremental image:
# the JLL initializers have not run in that process.
module JuMPyTrimDispatch

import GenOpt
import HiGHS
import JuMP
import JuMPyHiGHS
import MathOptInterface

const ITERATOR_TUPLES = (
    Tuple{Int64},
    Tuple{Float64},
    Tuple{Int64,Int64},
    Tuple{Int64,Float64},
    Tuple{Float64,Int64},
    Tuple{Float64,Float64},
)

function _supplement_iterator_roots!(roots::Vector{String}, signature)
    signature isa DataType || return nothing
    F = first(signature.parameters)
    F isa DataType &&
    parentmodule(F) === GenOpt &&
    startswith(string(nameof(F)), "#append_iterator#_merge_exprs") || return nothing
    # JIT inlining can hide some calls from --trace-dispatch. Reuse the
    # observed closure type, whose generated name varies with dependencies,
    # for the finite domains accepted by this image.
    for T in ITERATOR_TUPLES
        root = Tuple{F,GenOpt.Iterator{T}}
        push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
    end
    return nothing
end

function _supplement_generator_roots!(roots::Vector{String}, signature)
    signature isa DataType || return nothing
    for A in signature.parameters
        A isa DataType && A <: Base.Generator || continue
        F = A.parameters[2]
        F isa DataType &&
        parentmodule(F) === GenOpt &&
        startswith(string(nameof(F)), "#_new_values") || continue
        # map's abstract Function path reaches _collect dynamically, even
        # when ordinary Julia inlines both the closure and collection loop.
        for T in ITERATOR_TUPLES
            for root in (
                Tuple{F,T},
                Tuple{
                    typeof(Base._collect),
                    Vector{T},
                    Base.Generator{Vector{T},F},
                    Base.EltypeUnknown,
                    Base.HasShape{1},
                },
            )
                push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
            end
        end
    end
    return nothing
end

function _supplement_collect_roots!(roots::Vector{String}, signature)
    signature isa DataType && length(signature.parameters) == 5 || return nothing
    first(signature.parameters) === typeof(Base.collect_to_with_first!) || return nothing
    _, destination, _, iterator, state = signature.parameters
    # Julia can inline the first iteration while later collection dynamically
    # dispatches to the same iterator's steady-state loop.
    root = Tuple{typeof(Base.collect_to!),destination,iterator,Int64,state}
    push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
    return nothing
end

function _native_dispatch_domain(T)
    T isa DataType && T <: Type && length(T.parameters) == 1 || return nothing
    domain = only(T.parameters)
    domain isa Type && domain <: Union{JuMPyHiGHS.NativeNode,JuMPyHiGHS.NativeSet} ||
        return nothing
    return domain
end

function _supplement_native_dispatch_roots!(
    roots::Vector{String},
    signature,
    seen::Set{Any} = Set{Any}(),
)
    signature isa DataType || return nothing
    parameters = signature.parameters
    length(parameters) in (4, 6) || return nothing
    first(parameters) === typeof(JuMPyHiGHS._dispatch_native) || return nothing
    arity = (length(parameters) - 2) ÷ 2
    domain_types = Tuple(parameters[(arity+3):end])
    domains = map(_native_dispatch_domain, domain_types)
    any(isnothing, domains) && return nothing
    family = (parameters[2], domain_types...)
    family in seen && return nothing
    push!(seen, family)
    # Retain the generated wrapper for unobserved members of its finite ABI.
    # Retaining its callees does not retain these concrete method instances.
    for values in Iterators.product(map(Base.uniontypes, domains)...)
        root = Tuple{first(parameters),parameters[2],values...,domain_types...}
        push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
    end
    return nothing
end

function _is_backend_signature(line::String)
    occursin(
        r"Main\.|Test\.|Base\.Compiler|CachingOptimizer|\.Bridges\.|FunctionGeneratorBridge|JuMPyModel\.model\)|JuMP\.Containers\.|JuMP\._finalize_macro|Base\.Experimental\.entrypoint",
        line,
    ) && return false
    occursin(
        r"(?:JuMPyHiGHS|GenOpt|HiGHS|JuMP|MathOptInterface|MutableArithmetics)\.",
        line,
    ) && return true
    # Heterogeneous generator and nonlinear leaves also dispatch through Base
    # collection methods whose signatures do not mention an owning package.
    return occursin(
        r"typeof\(Base\.(?:_array_for(?:_inner)?|iterate|length|eachindex|axes|first|all|_all|setindex!|getindex)\)|Type\{Base\.(?:IteratorSize|LinearIndices)",
        line,
    ) && occursin(r"Array\{(?:Any|Real|Float64|Int64)|(?:UnitRange|OneTo)\{Int64\}", line)
end

function dispatch_roots(paths::Vector{String})
    roots = String[]
    native_dispatches = Set{Any}()
    for path in paths
        for (line_number, original) in enumerate(eachline(path))
            startswith(original, "precompile(Tuple{") || continue
            _is_backend_signature(original) || continue
            line = replace(original, "MutableArithmetics." => "JuMP._MA.")
            try
                expression = Meta.parse(line)
                expression isa Expr &&
                expression.head == :call &&
                expression.args[1] == :precompile &&
                length(expression.args) == 2 || error("Expected a precompile signature")
                signature = Core.eval(@__MODULE__, expression.args[2])
                signature isa Type && signature <: Tuple || error("Expected a tuple type")
                _supplement_iterator_roots!(roots, signature)
                _supplement_generator_roots!(roots, signature)
                _supplement_collect_roots!(roots, signature)
                _supplement_native_dispatch_roots!(roots, signature, native_dispatches)
            catch err
                error(
                    "Cannot resolve dispatch signature at $path:$line_number. " *
                    "Regenerate the trace with the image's dependency environment. " *
                    sprint(showerror, err),
                )
            end
            push!(roots, replace(line, r"^precompile\(" => "Base.Experimental.entrypoint("))
        end
    end
    return sort!(unique!(roots))
end

function generate(output::String, paths::Vector{String})
    roots = dispatch_roots(paths)
    isempty(roots) && error("The traces contain no backend dispatch signatures")
    open(output, "w") do io
        println(io, "#! format: off")
        println(io, "# Generated by generate_trim_dispatch.jl; do not edit.")
        println(io, "# Regenerate with the image's Julia and dependency versions.")
        println(io, "# Roots retain observed methods, not an ABI safety guarantee.")
        foreach(line -> println(io, line), roots)
    end
    return length(roots)
end

function main(args::Vector{String})
    length(args) >= 2 || error("Usage: generate_trim_dispatch.jl OUTPUT TRACE [TRACE ...]")
    count = generate(first(args), args[2:end])
    println("Wrote ", count, " supplementary roots to ", first(args))
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    JuMPyTrimDispatch.main(ARGS)
end
