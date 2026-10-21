# The opaque C handles deliberately erase Julia types. Retain the ordinary
# JuMP/GenOpt methods reachable through this finite backend's public operators;
# these are compilation roots, not alternative expression implementations.
# Use --trim=unsafe-warn: dependency internals still contain dynamic dispatch.
let J = JuMPyHiGHS, D = JuMPyHiGHS.JuMPyDirect, MOI = MathOptInterface
    V = J.NativeVariable
    scalar_types = (Float64, Int64, V, J.NativeAffine, J.NativeQuadratic, J.NativeNonlinear)
    template_types = (
        (GenOpt.ExprTemplate{E,V} for E in scalar_types)...,
        GenOpt.ExprTemplate{Int64,JuMP.VariableRef},
        GenOpt.ExprTemplate{Float64,JuMP.VariableRef},
    )
    node_types = (scalar_types..., template_types..., GenOpt.IteratorValues)
    set_types = Base.uniontypes(J.NativeSet)

    # JuMPyModel.apply resolves Base operators by name and unboxes Any values.
    for op in (+, -, *, /, ^), A in node_types, B in node_types
        if hasmethod(op, Tuple{A,B})
            Base.Experimental.entrypoint(op, (A, B))
        end
    end
    for op in (+, -, *, /, ^), A in scalar_types, B in scalar_types
        Base.Experimental.entrypoint(
            JuMP._MA.promote_operation, (typeof(op), Type{A}, Type{B}),
        )
        Base.Experimental.entrypoint(
            JuMP._MA.promote_operation_fallback, (typeof(op), Type{A}, Type{B}),
        )
    end
    for op in (+, -, sin, cos, exp, log, sqrt, abs), A in node_types
        if hasmethod(op, Tuple{A})
            Base.Experimental.entrypoint(op, (A,))
        end
    end
    for op in (+, -, sin, cos, exp, log, sqrt, abs), A in scalar_types
        Base.Experimental.entrypoint(JuMP._MA.promote_operation, (typeof(op), Type{A}))
        Base.Experimental.entrypoint(JuMP._MA.promote_operation_fallback, (typeof(op), Type{A}))
    end
    for A in (Vector{V}, Vector{Float64}), I in (
        Int64, GenOpt.IteratorValues, template_types...,
    )
        if hasmethod(getindex, Tuple{A,I})
            Base.Experimental.entrypoint(getindex, (A, I))
        end
    end

    for F in (scalar_types..., template_types...), S in set_types
        Base.Experimental.entrypoint(D.add_constraint, (J.NativeModel, F, S))
    end
    for F in scalar_types
        Base.Experimental.entrypoint(D.set_objective_function, (J.NativeModel, F))
        Base.Experimental.entrypoint(D.value, (F,))
        Base.Experimental.entrypoint(JuMP.value, (typeof(D._variable_value), F))
        Base.Experimental.entrypoint(
            JuMP._evaluate_expr,
            (MOI.Nonlinear.OperatorRegistry, typeof(D._variable_value), F),
        )
    end
    Base.Experimental.entrypoint(Base.Iterators.only, (Tuple{J.NativeNonlinear},))
    for F in (V, J.NativeAffine, J.NativeQuadratic, J.NativeNonlinear)
        Base.Experimental.entrypoint(getproperty, (F, Symbol))
    end
    for T in (Float64, Int64, Bool)
        Base.Experimental.entrypoint(
            MOI.Nonlinear.eval_univariate_function,
            (MOI.Nonlinear.OperatorRegistry, Symbol, T),
        )
    end
    for op in (==, !=, <, <=, >, >=, isequal, Base.add_sum, Base.mul_prod),
        A in (Float64, Int64, Bool),
        B in (Float64, Int64, Bool)
        Base.Experimental.entrypoint(op, (A, B))
    end
    Base.Experimental.entrypoint(
        MOI.Nonlinear._is_registered, (MOI.Nonlinear.OperatorRegistry, Symbol, Int64),
    )
    Base.Experimental.entrypoint(
        MOI.Nonlinear.eval_multivariate_function,
        (MOI.Nonlinear.OperatorRegistry, Symbol, Vector{Real}),
    )
    Base.Experimental.entrypoint(haskey, (Dict{Symbol,Int64}, Symbol))
    # The C value entry point converts the result of dynamically dispatched
    # value(). Inference can otherwise fold away even Float64(::Float64).
    for T in (Float64, Int64, Bool)
        Base.Experimental.entrypoint(Float64, (T,))
        Base.Experimental.entrypoint(convert, (Type{Float64}, T))
    end

    # Native nonlinear expressions and generators contain heterogeneous leaves.
    # Retain each supported leaf at the recursive dynamic-dispatch boundaries.
    moi_types = (
        Float64, Int64, Bool, MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction, GenOpt.IteratorIndex,
        Vector{MOI.VariableIndex}, Vector{Float64}, Vector{Int64},
        GenOpt.ContiguousArrayOfVariables{1},
        GenOpt._DataArray{Float64,1,Vector{Float64}},
        Tuple{Int64}, Tuple{Float64}, Tuple{Int64,Float64},
        Tuple{Int64,Int64}, Tuple{Float64,Float64},
    )
    for F in moi_types
        Base.Experimental.entrypoint(GenOpt._expand, (F, Vector{Any}))
        Base.Experimental.entrypoint(GenOpt._is_numeric, (F,))
        Base.Experimental.entrypoint(setindex!, (Vector{Any}, F, Int64))
    end
    Base.Experimental.entrypoint(==, (MOI.VariableIndex, MOI.VariableIndex))
    Base.Experimental.entrypoint(isequal, (MOI.VariableIndex, MOI.VariableIndex))
    Base.Experimental.entrypoint(hash, (MOI.VariableIndex, UInt))
    for F in (scalar_types..., JuMP.GenericNonlinearExpr{JuMP.VariableRef},
              GenOpt.IteratorIndex,
              GenOpt.ArrayOfVariables{Float64,1,V,J.NativeModel},
              GenOpt._DataArray{Float64,1,Vector{Float64}})
        Base.Experimental.entrypoint(GenOpt._reindex, (F, Vector{Int64}))
        Base.Experimental.entrypoint(GenOpt._check_belongs_to_model, (F, J.NativeModel))
        Base.Experimental.entrypoint(JuMP._throw_if_not_real, (F,))
        Base.Experimental.entrypoint(JuMP.moi_function, (Nothing, F))
        Base.Experimental.entrypoint(JuMP.moi_function, (J.NativeModel, F))
    end
    for F in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64},
              MOI.ScalarQuadraticFunction{Float64}, MOI.ScalarNonlinearFunction),
        S in set_types
        Base.Experimental.entrypoint(
            MOI.Utilities.normalize_and_add_constraint, (J.Optimizer, F, S),
        )
    end
    polynomial_types = (
        Float64, Int64, Bool, MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64},
    )
    for F in (MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}),
        G in polynomial_types
        Base.Experimental.entrypoint(convert, (Type{F}, G))
    end
    Base.Experimental.entrypoint(
        convert, (Type{MOI.ScalarNonlinearFunction}, MOI.ScalarNonlinearFunction),
    )
    for Q in (Missing, MOI.ScalarQuadraticFunction{Float64}), F in polynomial_types
        Base.Experimental.entrypoint(
            MOI.Nonlinear.SymbolicAD._add_to_quadratic!, (Q, Float64, F),
        )
    end
    for head in (:+, :-, :*, :/, :^, :sin, :cos, :exp, :log, :sqrt, :abs)
        Base.Experimental.entrypoint(
            MOI.Nonlinear.SymbolicAD.simplify!,
            (Val{head}, MOI.ScalarNonlinearFunction),
        )
    end

    # Generator expansion and nonlinear arguments use Vector{Any} internally.
    for f in (length, isempty, eachindex, copy, similar, iterate)
        Base.Experimental.entrypoint(f, (Vector{Any},))
    end
    for T in (Tuple{Int64}, Tuple{Float64}, Tuple{Int64,Int64},
              Tuple{Int64,Float64}, Tuple{Float64,Float64})
        Base.Experimental.entrypoint(getindex, (Vector{T}, Int64))
        Base.Experimental.entrypoint(getindex, (T, Int64))
        Base.Experimental.entrypoint(length, (Vector{T},))
    end
    Base.Experimental.entrypoint(iterate, (Vector{Any}, Int64))
    Base.Experimental.entrypoint(iterate, (Base.OneTo{Int64},))
    Base.Experimental.entrypoint(iterate, (Base.OneTo{Int64}, Int64))
    Base.Experimental.entrypoint(getindex, (Vector{Any}, Int64))
    Base.Experimental.entrypoint(HiGHS.Highs_destroy, (J.Optimizer,))
    Base.Experimental.entrypoint(HiGHS.HighsCallbackDataIn, (Cint,))
    # HiGHS 1.24.1's default interrupt callback (not a user callback). This
    # generated name is dependency-version-specific, like the dispatch trace.
    Base.Experimental.entrypoint(Tuple{HiGHS.var"#45#46", Int32, Ptr{Int8}, HiGHS.HighsCallbackDataOut})
end
