"""
    TrimDispatch

Generate supplementary compilation roots of a backend image from
`--trace-dispatch` traces of ordinary Julia runs of its workloads and tests,
using the exact Julia version and dependency environment of the image build.
See [`generate`](@ref).

Do not execute solver workloads while generating a nonincremental image:
the JLL initializers have not run in that process.
"""
module TrimDispatch

import ..JuMPy: GenOpt

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
        # map's abstract Function path reaches the Generator constructor and
        # _collect dynamically, even when ordinary Julia inlines both calls.
        for T in ITERATOR_TUPLES
            for root in (
                Tuple{F,T},
                Tuple{Type{Base.Generator},F,Vector{T}},
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

function _supplement_storage_roots!(
    roots::Vector{String},
    signature,
    seen::Set{Any} = Set{Any}(),
)
    signature isa DataType || return nothing
    for A in signature.parameters
        A isa DataType && A <: Vector || continue
        T = eltype(A)
        isconcretetype(T) && T <: Tuple || continue
        T in seen && continue
        push!(seen, T)
        # Data lookups append columns to iterator tuples. Retain storage for
        # observed shapes instead of enumerating every possible tuple arity.
        for root in (
            Tuple{Type{A},UndefInitializer,Int64},
            Tuple{Type{A},UndefInitializer,Tuple{Int64}},
            Tuple{
                typeof(Base._array_for_inner),
                Type{T},
                Base.HasShape{1},
                Tuple{Base.OneTo{Int64}},
            },
            Tuple{typeof(setindex!),A,T,Int64},
            Tuple{typeof(GenOpt._arity),GenOpt.Iterator{T}},
        )
            push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
        end
        for C in (A, T)
            for f in (length, isempty, eachindex, first, last, iterate)
                root = Tuple{typeof(f),C}
                push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
            end
            for f in (getindex, iterate)
                root = Tuple{typeof(f),C,Int64}
                push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
            end
        end
        for f in (copy, similar)
            root = Tuple{typeof(f),A}
            push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
        end
        for S in (Int64, Float64, Bool)
            root = Tuple{typeof(Base.setindex_widen_up_to),A,S,Int64}
            push!(roots, "Base.Experimental.entrypoint(" * repr(root) * ")")
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

function _native_dispatch_domain(J::Module, T)
    T isa DataType && T <: Type && length(T.parameters) == 1 || return nothing
    domain = only(T.parameters)
    domain isa Type && domain <: Union{J.NativeNode,J.NativeSet} || return nothing
    return domain
end

function _supplement_native_dispatch_roots!(
    roots::Vector{String},
    J::Module,
    signature,
    seen::Set{Any} = Set{Any}(),
)
    signature isa DataType || return nothing
    parameters = signature.parameters
    length(parameters) in (4, 6) || return nothing
    first(parameters) === typeof(J._dispatch_native) || return nothing
    arity = (length(parameters) - 2) ÷ 2
    domain_types = Tuple(parameters[(arity+3):end])
    domains = map(T -> _native_dispatch_domain(J, T), domain_types)
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

# Whether the traced `line` is retained: calls of the test harness and the
# compiler, and of JuMP's cached models, which the backend does not use, are
# not.
function _is_backend_signature(line::String)
    return !occursin(
        r"Main\.|Tests?\.|Base\.Compiler|\.Bridges\.|FunctionGeneratorBridge|JuMPyModel\.model\)|JuMP\.Containers\.|JuMP\._finalize_macro|Base\.Experimental\.entrypoint|CachingOptimizer",
        line,
    )
end

# A module in which every loaded package is bound to its name, to evaluate
# the fully qualified signatures of traces.
function _resolver()
    resolver = Module()
    for (_, M) in Base.loaded_modules
        Core.eval(resolver, :(const $(nameof(M)) = $M))
    end
    return resolver
end

# The packages mentioned in a traced `line` that are not loaded, such as
# another backend whose calls appear in shared test traces.
function _unloaded_packages(line::String, resolver::Module)
    names = Set(m.captures[1] for m in eachmatch(r"(?<![\w.])([A-Za-z_]\w*)\.", line))
    return filter(name -> !isdefined(resolver, Symbol(name)), names)
end

function dispatch_roots(paths::Vector{String}, J::Module)
    resolver = _resolver()
    roots = String[]
    native_dispatches = Set{Any}()
    storage_types = Set{Any}()
    for path in paths
        for (line_number, line) in enumerate(eachline(path))
            startswith(line, "precompile(Tuple{") || continue
            _is_backend_signature(line) || continue
            isempty(_unloaded_packages(line, resolver)) || continue
            try
                expression = Meta.parse(line)
                expression isa Expr &&
                expression.head == :call &&
                expression.args[1] == :precompile &&
                length(expression.args) == 2 || error("Expected a precompile signature")
                signature = Core.eval(resolver, expression.args[2])
                signature isa Type && signature <: Tuple || error("Expected a tuple type")
                _supplement_iterator_roots!(roots, signature)
                _supplement_generator_roots!(roots, signature)
                _supplement_collect_roots!(roots, signature)
                _supplement_storage_roots!(roots, signature, storage_types)
                _supplement_native_dispatch_roots!(roots, J, signature, native_dispatches)
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

"""
    generate(output, traces, J::Module)

Write to `output` the roots of the calls traced in the files `traces` of the
workloads of the backend module `J`. Calls of packages that are not loaded,
such as another backend in shared test traces, are skipped. Register them with
[`add_roots`](@ref) in the entry script of the image build.
"""
function generate(output::String, traces::Vector{String}, J::Module)
    roots = dispatch_roots(traces, J)
    isempty(roots) && error("The traces contain no backend dispatch signatures")
    open(output, "w") do io
        println(io, "#! format: off")
        println(io, "# Generated by JuMPy.TrimDispatch; do not edit.")
        println(io, "# Regenerate with the image's Julia and dependency versions.")
        println(io, "# Roots retain observed methods, not an ABI safety guarantee.")
        foreach(line -> println(io, line), roots)
    end
    return length(roots)
end

"""
    add_roots(path)

Register the roots written by [`generate`](@ref) to `path`. The signatures are
evaluated in a module where each loaded package is bound to its name.
"""
function add_roots(path::String)
    resolver = _resolver()
    for line in eachline(path)
        startswith(line, "Base.Experimental.entrypoint(") || continue
        Core.eval(resolver, Meta.parse(line))
    end
    return
end

end
