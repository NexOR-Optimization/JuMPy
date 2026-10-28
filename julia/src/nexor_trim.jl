# Keep solver options in a fixed dictionary type. The upstream untyped
# generator chooses types from runtime values, which trimming cannot retain.
# Remove this compatibility method once NexOR uses a typed dictionary.
function NexOR.JSON.lower(solver::NexOR.OptimizerWithAttributes)
    parameters = Dict{String,Any}()
    for (attr, value) in solver.params
        parameters[NexOR._parameter_name(attr)] = value
    end
    return Dict{String,Any}(
        "name" => NexOR.solver_id(solver.optimizer),
        "parameters" => parameters,
    )
end

# Avoid the predicate closure used by the upstream enum lookup.
function NexOR._status(::Type{E}, name::String) where {E}
    for status in instances(E)
        if string(status) == name
            return status
        end
    end
    throw(ArgumentError("unknown status: $name"))
end

# Solver namespaces add one inferred dictionary layer inside Any-valued
# options. Normalize it to JSON's statically supported dictionary shape.
function NexOR.JSON.applyany(
    style::NexOR.JSON.JSONWriteStyle,
    f::F,
    key::K,
    @nospecialize(value),
) where {F,K}
    if value isa Dict{String,Dict{String,Any}}
        object = Dict{String,Any}()
        for (name, options) in value
            object[name] = options
        end
        return NexOR.JSON._writenested(f, key, object)
    end
    return invoke(
        NexOR.JSON.applyany,
        Tuple{NexOR.JSON.JSONStyle,Any,Any,Any},
        style,
        f,
        key,
        value,
    )
end

# MockOptimizer builds these dictionaries through a runtime coefficient
# type. Retain the concrete Float64 constructors in the trimmed library.
for T in (
    Dict{Int,Float64},
    Dict{MOI.VariableIndex,Dict{Int,Float64}},
    Dict{MOI.VariableIndex,Float64},
)
    Base.Experimental.entrypoint(T, ())
end

# The parametric constructor is selected dynamically by MockOptimizer.
let mock = NexOR.Optimizer().model
    types = Tuple(typeof(getfield(mock, i)) for i = 1:fieldcount(typeof(mock)))
    Base.Experimental.entrypoint(typeof(mock), types)
    # MOI copies functions and sets through abstractly typed containers.
    # Retain the concrete scalar combinations exposed by the C ABI.
    for F in (
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    )
        for S in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64})
            Base.Experimental.entrypoint(
                MOI.Utilities._rethrow_if_not_NotAllowedError,
                (MOI.AddConstraintNotAllowed{F,S},),
            )
            Base.Experimental.entrypoint(
                MOI.Utilities.normalize_and_add_constraint,
                (Optimizer, F, S),
            )
            for M in (Optimizer, typeof(mock), MOI.FileFormats.MOF.Model{Float64})
                Base.Experimental.entrypoint(MOI.add_constraint, (M, F, S))
                Base.Experimental.entrypoint(MOI.supports_constraint, (M, Type{F}, Type{S}))
            end
            CI = MOI.ConstraintIndex{F,S}
            Base.Experimental.entrypoint(CI, (Int,))
            Base.Experimental.entrypoint(Pair{CI,CI}, (CI, CI))
            Base.Experimental.entrypoint(hash, (CI, UInt))
            Base.Experimental.entrypoint(==, (CI, CI))
            Base.Experimental.entrypoint(
                MOI.get,
                (MOI.FileFormats.MOF.Model{Float64}, MOI.ListOfConstraintIndices{F,S}),
            )
        end
        Base.Experimental.entrypoint(MOI.Utilities.map_indices, (MOI.Utilities.IndexMap, F))
        for M in (Optimizer, typeof(mock), MOI.FileFormats.MOF.Model{Float64})
            Base.Experimental.entrypoint(MOI.set, (M, MOI.ObjectiveFunction{F}, F))
        end
    end
    for S in (MOI.ZeroOne, MOI.Integer)
        Base.Experimental.entrypoint(
            MOI.Utilities._rethrow_if_not_NotAllowedError,
            (MOI.AddConstraintNotAllowed{MOI.VariableIndex,S},),
        )
        Base.Experimental.entrypoint(
            MOI.Utilities.normalize_and_add_constraint,
            (Optimizer, MOI.VariableIndex, S),
        )
        for M in (Optimizer, typeof(mock), MOI.FileFormats.MOF.Model{Float64})
            Base.Experimental.entrypoint(MOI.add_constraint, (M, MOI.VariableIndex, S))
        end
        Base.Experimental.entrypoint(MOI.ConstraintIndex{MOI.VariableIndex,S}, (Int,))
    end
    for (A, T, V) in (
        (MOI.TerminationStatus, String, MOI.TerminationStatusCode),
        (MOI.PrimalStatus, String, MOI.ResultStatusCode),
        (MOI.DualStatus, String, MOI.ResultStatusCode),
        (MOI.ResultCount, Int, Int),
        (MOI.ObjectiveValue, Float64, Float64),
        (MOI.DualObjectiveValue, Float64, Float64),
    )
        Base.Experimental.entrypoint(NexOR._lift, (A, T))
        Base.Experimental.entrypoint(MOI.set, (typeof(mock), A, V))
    end
end

# Incremental edits after a solve reset NexOR's non-incremental optimizer.
Base.Experimental.entrypoint(
    MOI.Utilities._rethrow_if_not_NotAllowedError,
    (MOI.AddVariableNotAllowed,),
)
Base.Experimental.entrypoint(
    MOI.Utilities._rethrow_if_not_NotAllowedError,
    (MOI.SetAttributeNotAllowed{MOI.ObjectiveSense},),
)
for F in (
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
)
    Base.Experimental.entrypoint(
        MOI.Utilities._rethrow_if_not_NotAllowedError,
        (MOI.SetAttributeNotAllowed{MOI.ObjectiveFunction{F}},),
    )
end

for A in (MOI.Silent, MOI.TimeLimitSec, MOI.NumberOfThreads, MOI.RawOptimizerAttribute)
    Base.Experimental.entrypoint(NexOR._parameter_name, (A,))
end

# SymbolicAD dispatches on runtime operator symbols and heterogeneous nodes.
for head in union(
    MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS,
    MOI.Nonlinear.DEFAULT_MULTIVARIATE_OPERATORS,
)
    Base.Experimental.entrypoint(Val{head}, ())
    Base.Experimental.entrypoint(
        MOI.Nonlinear.SymbolicAD.simplify!,
        (Val{head}, MOI.ScalarNonlinearFunction),
    )
end
for F in (
    Float64,
    Bool,
    Int,
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
    MOI.ScalarAffineTerm{Float64},
    MOI.ScalarQuadraticTerm{Float64},
)
    for f in (
        MOI.Nonlinear.SymbolicAD._eval_if_constant,
        MOI.Nonlinear.SymbolicAD._simplify_if_quadratic!,
        MOI.Nonlinear.SymbolicAD._iszero,
        MOI.Nonlinear.SymbolicAD._isone,
        MOI.Nonlinear.SymbolicAD._isnum,
    )
        Base.Experimental.entrypoint(f, (F,))
    end
    Base.Experimental.entrypoint(MOI.Nonlinear.SymbolicAD._isexpr, (F, Symbol))
    Base.Experimental.entrypoint(MOI.Nonlinear.SymbolicAD._isexpr, (F, Symbol, Int))
    for R in (Missing, MOI.ScalarQuadraticFunction{Float64})
        Base.Experimental.entrypoint(
            MOI.Nonlinear.SymbolicAD._add_to_quadratic!,
            (R, Float64, F),
        )
    end
end
for R in (Missing, MOI.ScalarQuadraticFunction{Float64})
    for F in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64})
        for G in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64})
            Base.Experimental.entrypoint(
                MOI.Nonlinear.SymbolicAD._add_to_quadratic!,
                (R, Float64, F, G),
            )
        end
    end
end
for T in
    (Tuple{MOI.ScalarNonlinearFunction}, Vector{Any}, Vector{MOI.ScalarAffineTerm{Float64}})
    Base.Experimental.entrypoint(only, (T,))
end

for F in (
    MOI.VariableIndex,
    MOI.ScalarNonlinearFunction,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarAffineTerm{Float64},
    MOI.ScalarQuadraticTerm{Float64},
)
    Base.Experimental.entrypoint(getproperty, (F, Symbol))
end
for F in (Float64, Bool, MOI.VariableIndex, MOI.ScalarNonlinearFunction)
    for G in (Float64, Bool, MOI.VariableIndex, MOI.ScalarNonlinearFunction)
        Base.Experimental.entrypoint(==, (F, G))
    end
end

# Dynamic node fields in SymbolicAD contain these concrete arrays.
for T in (Vector{Any}, Vector{Float64})
    for f in (
        length,
        eachindex,
        firstindex,
        lastindex,
        axes,
        size,
        iterate,
        isempty,
        only,
        first,
        last,
        copy,
        pop!,
        empty!,
    )
        Base.Experimental.entrypoint(f, (T,))
    end
    Base.Experimental.entrypoint(iterate, (T, Int))
    Base.Experimental.entrypoint(getindex, (T, Int))
    Base.Experimental.entrypoint(resize!, (T, Int))
    Base.Experimental.entrypoint(copyto!, (T, T))
    Base.Experimental.entrypoint(append!, (T, T))
end
for T in (
    Float64,
    Bool,
    Int,
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
    Tuple{MOI.ScalarNonlinearFunction},
)
    Base.Experimental.entrypoint(push!, (Vector{Any}, T))
    Base.Experimental.entrypoint(setindex!, (Vector{Any}, T, Int))
end

for T in (Base.OneTo{Int}, UnitRange{Int})
    Base.Experimental.entrypoint(iterate, (T,))
    Base.Experimental.entrypoint(iterate, (T, Int))
end

Base.Experimental.entrypoint(MOI.supports, (NexOR.Optimizer, MOI.Silent))
Base.Experimental.entrypoint(MOI.get, (ScalarModel{Float64}, MOI.Silent))
Base.Experimental.entrypoint(
    MOI.Utilities.map_indices,
    (MOI.Utilities.IndexMap, MOI.Silent, Bool),
)
Base.Experimental.entrypoint(Float64, (Float64,))
Base.Experimental.entrypoint(Float64, (Int,))

# NexOR's solver-option dictionaries infer their value types at runtime.
for V in (
    Bool,
    Int,
    Float64,
    String,
    Dict{String,Bool},
    Dict{String,Dict{String,Bool}},
    Dict{String,Dict{String,Any}},
    Dict{String,Any},
)
    Base.Experimental.entrypoint(Pair, (String, V))
    Base.Experimental.entrypoint(Dict, (Pair{String,V},))
    Base.Experimental.entrypoint(Dict{String,Any}, (Pair{String,V},))
    D = Dict{String,V}
    Base.Experimental.entrypoint(iterate, (D,))
    Base.Experimental.entrypoint(iterate, (D, Int))
    Base.Experimental.entrypoint(length, (D,))
    Base.Experimental.entrypoint(pairs, (D,))
    Base.Experimental.entrypoint(get, (D, String, Nothing))
end


for V in (
    Bool,
    Int,
    Float64,
    String,
    Dict{String,Bool},
    Dict{String,Dict{String,Bool}},
    Dict{String,Dict{String,Any}},
    Dict{String,Any},
)
    Base.Experimental.entrypoint(empty, (Dict{Any,Any}, Type{String}, Type{V}))
end
let
    keys = (:api_version, :problem_type, :solver, :options, :problem)
    problem_type = typeof(NexOR.JSON.parse("{}"))
    fields = Tuple{String,String,String,Dict{String,Any},problem_type}
    envelope = NamedTuple{keys,fields}
    Base.Experimental.entrypoint(NamedTuple{keys}, (fields,))
    Base.Experimental.entrypoint(NexOR.JSON.json, (envelope,))
end

for D in (Dict{String,Any}, Dict{String,Dict{String,Any}})
    Base.Experimental.entrypoint(NexOR.JSON.json, (D,))
end

for D in (Dict{String,Any}, Dict{Any,Any})
    for f in (length, iterate, keys, values, pairs, isempty)
        Base.Experimental.entrypoint(f, (D,))
    end
    Base.Experimental.entrypoint(iterate, (D, Int))
    Base.Experimental.entrypoint(get, (D, String, Nothing))
    Base.Experimental.entrypoint(get, (D, String, String))
    Base.Experimental.entrypoint(getindex, (D, String))
    Base.Experimental.entrypoint(haskey, (D, String))
end

# Nonlinear argument lists erase their leaf types during index mapping.
for M in
    (typeof(Base.Fix1(getindex, MOI.Utilities.IndexMap())), typeof(MOI.Utilities.xor_index))
    for T in (
        Float64,
        Bool,
        Int,
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    )
        Base.Experimental.entrypoint(MOI.Utilities.map_indices, (M, T))
    end
end

for T in (
    Float64,
    Bool,
    Int,
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
)
    for f in (
        MOI.FileFormats.MOF._convert_nonlinear_argument_to_mof,
        MOI.FileFormats.MOF._convert_nonlinear_to_mof,
    )
        Base.Experimental.entrypoint(f, (T, Vector{Any}, Dict{MOI.VariableIndex,String}))
    end
end
for T in (
    String,
    NamedTuple{(:type, :index),Tuple{String,Int}},
    NamedTuple{(:type, :args),Tuple{String,Vector{Any}}},
)
    Base.Experimental.entrypoint(push!, (Vector{Any}, T))
end

# Responses use JSON's untyped object and array containers.
let
    sample = NexOR.JSON.parse("{\"a\":1,\"b\":2}")
    D = typeof(sample)
    mock = NexOR.Optimizer().model
    Base.Experimental.entrypoint(NexOR.JSON.parse, (String,))
    Base.Experimental.entrypoint(NexOR.load_solution!, (typeof(mock), D))
    Base.Experimental.entrypoint(iterate, (D,))
    state = iterate(sample)[2]
    Base.Experimental.entrypoint(iterate, (D, typeof(state)))
    if hasmethod(iterate, Tuple{D,Nothing})
        Base.Experimental.entrypoint(iterate, (D, Nothing))
    end
    Base.Experimental.entrypoint(getindex, (D, String))
    Base.Experimental.entrypoint(haskey, (D, String))
    Base.Experimental.entrypoint(get, (D, String, Nothing))
    Base.Experimental.entrypoint(get, (D, String, String))
    Base.Experimental.entrypoint(in, (String, typeof(NexOR._TERMINAL_STATES)))
end
let Z = typeof(zip(MOI.VariableIndex[], Any[]))
    Base.Experimental.entrypoint(zip, (Vector{MOI.VariableIndex}, Vector{Any}))
    Base.Experimental.entrypoint(iterate, (Z,))
    Base.Experimental.entrypoint(iterate, (Z, Tuple{Int,Int}))
end
Base.Experimental.entrypoint(convert, (Type{Union{Nothing,String}}, String))

Base.Experimental.entrypoint(Base.indexed_iterate, (Pair{String,Any}, Int))
Base.Experimental.entrypoint(Base.indexed_iterate, (Pair{String,Any}, Int, Int))
Base.Experimental.entrypoint(String, (String,))
Base.Experimental.entrypoint(setproperty!, (NexOR.Optimizer, Symbol, String))
Base.Experimental.entrypoint(getindex, (typeof(NexOR._SOLUTION_ATTRIBUTES), String))

Base.Experimental.entrypoint(==, (String, String))
Base.Experimental.entrypoint(!=, (String, String))
Base.Experimental.entrypoint(!, (Bool,))
Base.Experimental.entrypoint(==, (MOI.Silent, MOI.Silent))
Base.Experimental.entrypoint(string, (MOI.TerminationStatusCode,))
Base.Experimental.entrypoint(string, (MOI.ResultStatusCode,))

let T = Tuple{Vector{MOI.VariableIndex},Vector{Any}}
    Base.Experimental.entrypoint(Base.Iterators.Zip, (T,))
    Base.Experimental.entrypoint(Base.Iterators.Zip{T}, (T,))
end
Base.Experimental.entrypoint(Base.indexed_iterate, (Tuple{MOI.VariableIndex,Float64}, Int))
Base.Experimental.entrypoint(
    Base.indexed_iterate,
    (Tuple{MOI.VariableIndex,Float64}, Int, Int),
)
Base.Experimental.entrypoint(
    MOI.set,
    (typeof(NexOR.Optimizer().model), MOI.VariablePrimal, MOI.VariableIndex, Float64),
)

let I = Tuple{Vector{MOI.VariableIndex},Vector{Any}}
    Base.Experimental.entrypoint(
        Base.Iterators._zip_iterate_all,
        (I, Tuple{Tuple{},Tuple{}}),
    )
    Base.Experimental.entrypoint(
        Base.Iterators._zip_iterate_all,
        (I, Tuple{Tuple{Int},Tuple{Int}}),
    )
end

Base.Experimental.entrypoint(sizeof, (MOI.VariablePrimal,))
Base.Experimental.entrypoint(propertynames, (MOI.VariablePrimal,))
Base.Experimental.entrypoint(hasproperty, (MOI.VariablePrimal, Symbol))
Base.Experimental.entrypoint(getproperty, (MOI.VariablePrimal, Symbol))
Base.Experimental.entrypoint(hash, (MOI.VariablePrimal, UInt))

Base.Experimental.entrypoint(length, (Core.SimpleVector,))
Base.Experimental.entrypoint(getindex, (Core.SimpleVector, Int))
Base.Experimental.entrypoint(iterate, (Core.SimpleVector,))
Base.Experimental.entrypoint(iterate, (Core.SimpleVector, Int))
Base.Experimental.entrypoint(in, (Symbol, Tuple{Symbol}))

# Re-copying a model leaves constraint containers allocated. Adding the
# replacement variables visits those containers through MOI's broadcaster.
let
    cache = ScalarModel{Float64}()
    mof = NexOR.Optimizer().model.inner_model.model
    pending = Any[typeof(cache.constraints), typeof(mof.constraints)]
    seen = Set{Any}()
    while !isempty(pending)
        for T in Base.uniontypes(pop!(pending))
            T in seen && continue
            push!(seen, T)
            if T <: MOI.Utilities.StructOfConstraints
                Base.Experimental.entrypoint(MOI.Utilities._add_variable, (T,))
                append!(pending, fieldtypes(T))
            elseif T <: MOI.Utilities.VectorOfConstraints
                Base.Experimental.entrypoint(MOI.Utilities._add_variable, (T,))
            end
        end
    end
end
