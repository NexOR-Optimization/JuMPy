# Extra finite dispatch roots for experimental trimmed images. This file is
# build-time only: it must not constrain the primary shared-runtime image.
for (f, args) in (
    (Base.IteratorSize, (UnitRange{Int64},)),
    (Base._array_for, (Type{Any}, UnitRange{Int64}, Base.HasShape{1})),
    (Base.LinearIndices, (Vector{Any},)),
    (Base.first, (LinearIndices{1,Tuple{Base.OneTo{Int64}}},)),
    (Base.iterate, (UnitRange{Int64},)),
    (Base.iterate, (UnitRange{Int64}, Int64)),
    (Base.iterate, (Base.OneTo{Int64},)),
    (Base.iterate, (Base.OneTo{Int64}, Int64)),
    (Base.all, (typeof(GenOpt._is_numeric), Vector{Float64})),
    (Base.all, (typeof(GenOpt._is_numeric), Vector{Any})),
    (Base._array_for_inner, (Type{Float64}, Base.HasShape{1}, Tuple{Base.OneTo{Int64}})),
    (Base._array_for_inner, (Type{Int64}, Base.HasShape{1}, Tuple{Base.OneTo{Int64}})),
    (Base._array_for_inner, (Type{Vector{Float64}}, Base.HasShape{1}, Tuple{Base.OneTo{Int64}})),
    (Base.getindex, (Vector{Any}, UnitRange{Int64})),
    (MOI.Nonlinear.SymbolicAD._add_to_quadratic!, (MOI.ScalarQuadraticFunction{Float64}, Float64, MOI.VariableIndex)),
    (MOI.Nonlinear.SymbolicAD._add_to_quadratic!, (MOI.ScalarQuadraticFunction{Float64}, Float64, MOI.ScalarNonlinearFunction)),
    (Base.all, (typeof(GenOpt._is_numeric), Vector{MOI.VariableIndex})),
    (Base.all, (typeof(GenOpt._is_numeric), Vector{MOI.ScalarNonlinearFunction})),
    (Base._all, (typeof(GenOpt._is_numeric), Vector{MOI.VariableIndex}, Colon)),
    (Base._all, (typeof(GenOpt._is_numeric), Vector{MOI.ScalarNonlinearFunction}, Colon)),
    (Base.getindex, (Vector{Float64}, Int64)),
    (HiGHS.HighsCallbackDataIn, (Cint,)),
    (MOI.ScalarNonlinearFunction, (Symbol, Vector{MOI.VariableIndex})),
    (MOI.ScalarNonlinearFunction, (Symbol, Vector{MOI.ScalarNonlinearFunction})),
    (MOI.ScalarNonlinearFunction, (Symbol, Vector{MOI.AbstractScalarFunction})),
    (Base.all, (typeof(GenOpt._is_numeric), Vector{MOI.AbstractScalarFunction})),
    (Base._all, (typeof(GenOpt._is_numeric), Vector{MOI.AbstractScalarFunction}, Colon)),
    (Base.setindex_widen_up_to, (Vector{MOI.AbstractScalarFunction}, Float64, Int64)),
    (==, (MOI.VariableIndex, MOI.VariableIndex)),
    (MOI.Nonlinear.SymbolicAD._eval_if_constant, (Bool,)),
    (MOI.Nonlinear.SymbolicAD._iszero, (Bool,)),
    (MOI.Nonlinear.SymbolicAD._isone, (Bool,)),
    (Base.copy, (Bool,)),
    (Base.setindex_widen_up_to, (Vector{Vector{Float64}}, Float64, Int64)),
)
    Base.Experimental.entrypoint(f, args)
end
for f in (+, -, *, /, ^, ==, <, <=, >, >=)
    Base.Experimental.entrypoint(f, (Float64, Float64))
end
Base.Experimental.entrypoint(-, (Float64,))
for F in (Float64, MOI.VariableIndex, MOI.ScalarAffineFunction{Float64}, MOI.ScalarNonlinearFunction),
    S in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.ZeroOne, MOI.Integer)
    Base.Experimental.entrypoint(JuMPyHiGHS._add_set, (JuMPyHiGHS.Optimizer, F, S))
end
for T in (Float64, MOI.VariableIndex, MOI.ScalarNonlinearFunction)
    Base.Experimental.entrypoint(Base.setindex!, (Vector{Any}, T, Int64))
    Base.Experimental.entrypoint(Base.push!, (Vector{Any}, T))
end
# GenOpt collect widens from the first argument's type when later args differ.
for T in (Float64, MOI.VariableIndex, MOI.ScalarNonlinearFunction, Vector{Float64}, GenOpt.ContiguousArrayOfVariables{1}),
    S in (Float64, MOI.VariableIndex, MOI.ScalarNonlinearFunction, Vector{Float64}, GenOpt.ContiguousArrayOfVariables{1})
    T === S && continue
    Base.Experimental.entrypoint(Base.setindex_widen_up_to, (Vector{T}, S, Int64))
end

# Anonymous closure names change with dependency and Julia versions. Discover
# only closures referenced by the two relevant methods, validate their shapes,
# and stop the experimental build if those implementations change unexpectedly.
let
    function closure_types(f, signature, owner)
        types = Set{Any}()
        function visit(node)
            if node isa GlobalRef && node.mod === owner && isdefined(owner, node.name)
                value = getfield(owner, node.name)
                if (value isa DataType || value isa UnionAll) && value <: Function
                    push!(types, value)
                end
            elseif node isa Expr
                foreach(visit, node.args)
            end
            return
        end
        for code in code_lowered(f, signature)
            foreach(visit, code.code)
        end
        return types
    end

    callback_signature = (Cint, Ptr{Cchar}, HiGHS.HighsCallbackDataOut)
    callbacks = Any[]
    for T in closure_types(MOI.optimize!, (HiGHS.Optimizer,), HiGHS)
        if T isa DataType && Base.issingletontype(T) && hasmethod(T.instance, Tuple{callback_signature...})
            push!(callbacks, T.instance)
        end
    end
    length(callbacks) == 1 || error("Experimental trimming: cannot identify the HiGHS default interrupt callback")
    Base.Experimental.entrypoint(only(callbacks), callback_signature)

    expanders = DataType[]
    signature = (MOI.ScalarNonlinearFunction, Vector{Float64})
    for T in closure_types(GenOpt._expand, signature, GenOpt)
        if T isa UnionAll && !(T.body isa UnionAll)
            concrete = try
                Core.apply_type(T, Vector{Float64})
            catch
                continue
            end
            if fieldtypes(concrete) == (Vector{Float64},)
                push!(expanders, concrete)
            end
        end
    end
    length(expanders) == 1 || error("Experimental trimming: cannot identify the GenOpt expansion generator")
    generator = Base.Generator{Vector{Any},only(expanders)}
    Base.Experimental.entrypoint(
        Base.collect_to!, (Vector{MOI.AbstractScalarFunction}, generator, Int64, Int64),
    )
end
