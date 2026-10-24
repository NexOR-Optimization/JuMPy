# The opaque C handles deliberately erase Julia types. Retain the ordinary
# JuMP/GenOpt methods reachable through this finite backend's public operators;
# these are compilation roots, not alternative expression implementations.
# Use --trim=unsafe-warn: dependency internals still contain dynamic dispatch.
let J = JuMPyHiGHS, D = JuMPyHiGHS.JuMPyDirect, MOI = MathOptInterface
    V = J.NativeVariable
    base_scalar_types =
        (Float64, Int64, V, J.NativeAffine, J.NativeQuadratic, J.NativeNonlinear)
    scalar_types = Tuple(Base.uniontypes(J.NativeScalar))
    template_types = (
        (GenOpt.ExprTemplate{E,V} for E in base_scalar_types)...,
        GenOpt.ExprTemplate{Int64,JuMP.VariableRef},
        GenOpt.ExprTemplate{Float64,JuMP.VariableRef},
    )
    node_types = (scalar_types..., template_types..., GenOpt.IteratorValues)
    set_types = Base.uniontypes(J.NativeSet)
    numbers = (Float64, Int64, Bool)
    tuples = (
        Tuple{Int64},
        Tuple{Float64},
        Tuple{Int64,Int64},
        Tuple{Int64,Float64},
        Tuple{Float64,Int64},
        Tuple{Float64,Float64},
    )
    polynomial_functions =
        (MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64})
    polynomial_wrappers = Tuple(GenOpt.PolynomialFunction{F} for F in polynomial_functions)
    sum_types = Tuple(
        GenOpt.SumGenerator{F} for
        F in (MOI.VariableIndex, polynomial_functions..., MOI.ScalarNonlinearFunction)
    )

    for T in template_types
        Base.Experimental.entrypoint(J.JuMPyModel.lazy_sum, (T,))
    end

    # JuMPyModel.apply resolves Base operators by name and unboxes Any values.
    for op in (+, -, *, /, ^), A in node_types, B in node_types
        if hasmethod(op, Tuple{A,B})
            Base.Experimental.entrypoint(op, (A, B))
        end
    end
    for op in (+, -, *, /, ^), A in scalar_types, B in scalar_types
        Base.Experimental.entrypoint(
            JuMP._MA.promote_operation,
            (typeof(op), Type{A}, Type{B}),
        )
        Base.Experimental.entrypoint(
            JuMP._MA.promote_operation_fallback,
            (typeof(op), Type{A}, Type{B}),
        )
    end
    for op in (+, -, sin, cos, exp, log, sqrt, abs), A in node_types
        if hasmethod(op, Tuple{A})
            Base.Experimental.entrypoint(op, (A,))
        end
    end
    for op in (+, -, sin, cos, exp, log, sqrt, abs), A in scalar_types
        Base.Experimental.entrypoint(JuMP._MA.promote_operation, (typeof(op), Type{A}))
        Base.Experimental.entrypoint(
            JuMP._MA.promote_operation_fallback,
            (typeof(op), Type{A}),
        )
    end
    for A in (Vector{V}, Vector{Float64}),
        I in (Int64, GenOpt.IteratorValues, template_types...)

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
        if hasmethod(JuMP.value, Tuple{typeof(D._variable_value),F})
            Base.Experimental.entrypoint(JuMP.value, (typeof(D._variable_value), F))
        end
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
    for op in (
            +,
            -,
            *,
            /,
            ^,
            ==,
            !=,
            <,
            <=,
            >,
            >=,
            isequal,
            promote,
            Base.add_sum,
            Base.mul_prod,
        ),
        A in numbers,
        B in numbers

        Base.Experimental.entrypoint(op, (A, B))
    end
    # Constant folding promotes boxed arguments and converts their tuple to
    # floating-point values before evaluating the nonlinear operator.
    for T in numbers
        Base.Experimental.entrypoint(float, (T,))
        Base.Experimental.entrypoint(promote, (T,))
    end
    numeric_tuples =
        ((Tuple{T} for T in numbers)..., (Tuple{A,B} for A in numbers for B in numbers)...)
    for T in numeric_tuples
        args = ntuple(i -> zero(fieldtype(T, i)), fieldcount(T))
        broadcast = Base.Broadcast.broadcasted(float, args)
        Base.Experimental.entrypoint(Base.Broadcast.broadcasted, (typeof(float), T))
        Base.Experimental.entrypoint(Base.Broadcast.materialize, (typeof(broadcast),))
    end
    for T in (Tuple{Float64}, Tuple{Float64,Float64})
        Base.Experimental.entrypoint(collect, (T,))
    end
    Base.Experimental.entrypoint(
        MOI.Nonlinear._is_registered,
        (MOI.Nonlinear.OperatorRegistry, Symbol, Int64),
    )
    Base.Experimental.entrypoint(
        MOI.Nonlinear.eval_multivariate_function,
        (MOI.Nonlinear.OperatorRegistry, Symbol, Vector{Real}),
    )
    Base.Experimental.entrypoint(
        MOI.Nonlinear.eval_multivariate_function,
        (MOI.Nonlinear.OperatorRegistry, Symbol, Vector{Float64}),
    )
    Base.Experimental.entrypoint(haskey, (Dict{Symbol,Int64}, Symbol))
    # The C value entry point converts the result of dynamically dispatched
    # value(). Inference can otherwise fold away even Float64(::Float64).
    for T in (Float64, Int64), S in numbers
        Base.Experimental.entrypoint(T, (S,))
        Base.Experimental.entrypoint(convert, (Type{T}, S))
    end

    # Native nonlinear expressions and generators contain heterogeneous leaves.
    # Retain each supported leaf at the recursive dynamic-dispatch boundaries.
    moi_types = (
        Float64,
        Int64,
        Bool,
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
        GenOpt.IteratorIndex,
        Vector{MOI.VariableIndex},
        Vector{Float64},
        Vector{Int64},
        GenOpt.ContiguousArrayOfVariables{1},
        GenOpt._DataArray{Float64,1,Vector{Float64}},
        tuples...,
        polynomial_wrappers...,
        sum_types...,
    )
    for F in moi_types
        Base.Experimental.entrypoint(GenOpt._expand, (F, Vector{Any}))
        Base.Experimental.entrypoint(GenOpt._is_numeric, (F,))
        Base.Experimental.entrypoint(setindex!, (Vector{Any}, F, Int64))
        Base.Experimental.entrypoint(hash, (F, UInt))
    end
    Base.Experimental.entrypoint(==, (MOI.VariableIndex, MOI.VariableIndex))
    Base.Experimental.entrypoint(isequal, (MOI.VariableIndex, MOI.VariableIndex))
    Base.Experimental.entrypoint(hash, (MOI.VariableIndex, UInt))
    julia_leaves = (
        scalar_types...,
        Bool,
        JuMP.GenericNonlinearExpr{JuMP.VariableRef},
        GenOpt.IteratorIndex,
        GenOpt.ArrayOfVariables{Float64,1,V,J.NativeModel},
        GenOpt._DataArray{Float64,1,Vector{Float64}},
    )
    for F in julia_leaves
        Base.Experimental.entrypoint(GenOpt._reindex, (F, Vector{Int64}))
        Base.Experimental.entrypoint(GenOpt._check_belongs_to_model, (F, J.NativeModel))
        Base.Experimental.entrypoint(JuMP._throw_if_not_real, (F,))
        Base.Experimental.entrypoint(JuMP.moi_function, (Nothing, F))
        Base.Experimental.entrypoint(JuMP.moi_function, (J.NativeModel, F))
        Base.Experimental.entrypoint(JuMP.moi_function, (F,))
        Base.Experimental.entrypoint(hash, (F, UInt))
    end
    # Subexpression cache collisions depend on process-local object hashes.
    cache_keys = (J.NativeNonlinear, JuMP.NonlinearExpr, WeakRef, Nothing)
    for op in (isequal, ==), A in cache_keys, B in cache_keys
        Base.Experimental.entrypoint(op, (A, B))
    end
    for T in (WeakRef, Nothing)
        Base.Experimental.entrypoint(hash, (T, UInt))
    end
    for f in (any, findfirst)
        Base.Experimental.entrypoint(f, (typeof(JuMP._has_variable_ref_type), Vector{Any}))
    end
    for F in (julia_leaves..., tuples...)
        Base.Experimental.entrypoint(JuMP._has_variable_ref_type, (F,))
    end
    for F in (
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
            MOI.ScalarNonlinearFunction,
        ),
        S in set_types

        Base.Experimental.entrypoint(
            MOI.Utilities.normalize_and_add_constraint,
            (J.Optimizer, F, S),
        )
    end
    polynomial_types = (
        Float64,
        Int64,
        Bool,
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    )
    for F in polynomial_functions, G in polynomial_types

        if hasmethod(convert, Tuple{Type{F},G})
            Base.Experimental.entrypoint(convert, (Type{F}, G))
        end
    end
    for F in polynomial_functions
        Base.Experimental.entrypoint(convert, (Type{F}, GenOpt.PolynomialFunction{F}))
        Base.Experimental.entrypoint(D._solver_function, (GenOpt.PolynomialFunction{F},))
    end
    # Polynomial builders recurse through boxed nonlinear and lazy-sum leaves.
    for F in (
        polynomial_types...,
        MOI.ScalarNonlinearFunction,
        GenOpt.IteratorIndex,
        polynomial_wrappers...,
        sum_types...,
    )
        Base.Experimental.entrypoint(
            GenOpt._expand_affine!,
            (MOI.ScalarAffineFunction{Float64}, F, Vector{Any}, Float64),
        )
        Base.Experimental.entrypoint(
            GenOpt._expand_quadratic!,
            (MOI.ScalarQuadraticFunction{Float64}, F, Vector{Any}, Float64),
        )
    end
    for F in (Int64, Float64, GenOpt.IteratorIndex, MOI.ScalarNonlinearFunction)
        Base.Experimental.entrypoint(GenOpt._eval_index, (F, Vector{Any}))
    end
    Base.Experimental.entrypoint(
        convert,
        (Type{MOI.ScalarNonlinearFunction}, MOI.ScalarNonlinearFunction),
    )
    for Q in (Missing, MOI.ScalarQuadraticFunction{Float64}), F in polynomial_types
        Base.Experimental.entrypoint(
            MOI.Nonlinear.SymbolicAD._add_to_quadratic!,
            (Q, Float64, F),
        )
    end
    for head in (:+, :-, :*, :/, :^, :sin, :cos, :exp, :log, :sqrt, :abs)
        Base.Experimental.entrypoint(
            MOI.Nonlinear.SymbolicAD.simplify!,
            (Val{head}, MOI.ScalarNonlinearFunction),
        )
    end

    # Generic collection loops are not always observed by dispatch tracing.
    for T in (tuples..., Any, Float64, Int64, Bool)
        array = Vector{T}
        Base.Experimental.entrypoint(array, (UndefInitializer, Int64))
        Base.Experimental.entrypoint(array, (UndefInitializer, Tuple{Int64}))
        for f in (length, isempty, eachindex, copy, similar, iterate)
            Base.Experimental.entrypoint(f, (array,))
        end
        Base.Experimental.entrypoint(iterate, (array, Int64))
        Base.Experimental.entrypoint(getindex, (array, Int64))
        Base.Experimental.entrypoint(setindex!, (array, T, Int64))
        Base.Experimental.entrypoint(any, (Function, array))
        Base.Experimental.entrypoint(Base._any, (Function, array, Colon))
    end
    for T in tuples
        Base.Experimental.entrypoint(getindex, (T, Int64))
        for f in (length, iterate, first, last, eachindex)
            Base.Experimental.entrypoint(f, (T,))
        end
        Base.Experimental.entrypoint(iterate, (T, Int64))
        Base.Experimental.entrypoint(map, (Function, Vector{T}))
        Base.Experimental.entrypoint(GenOpt._arity, (GenOpt.Iterator{T},))
    end
    for T in (tuples..., Any), S in numbers
        Base.Experimental.entrypoint(Base.setindex_widen_up_to, (Vector{T}, S, Int64))
    end
    for C in (tuples..., Vector{Any}, Vector{Int64}, Vector{Float64})
        Base.Experimental.entrypoint(
            GenOpt._getindex_fallback,
            (C, Vector{Any}, Vector{Any}),
        )
    end
    for N = 1:3
        dimensions = NTuple{N,Int64}
        indices = CartesianIndices{N,NTuple{N,Base.OneTo{Int64}}}
        index = CartesianIndex{N}
        Base.Experimental.entrypoint(CartesianIndices, (dimensions,))
        Base.Experimental.entrypoint(iterate, (indices,))
        Base.Experimental.entrypoint(iterate, (indices, index))
        Base.Experimental.entrypoint(getindex, (indices, Int64))
        Base.Experimental.entrypoint(getindex, (indices, ntuple(_ -> Int64, N)...))
        Base.Experimental.entrypoint(getindex, (index, Int64))
        Base.Experimental.entrypoint(
            Base._to_linear_index,
            (indices, ntuple(_ -> Int64, N)...),
        )
        for f in (length, size, axes, first, last, eachindex)
            Base.Experimental.entrypoint(f, (indices,))
        end
    end
    Base.Experimental.entrypoint(iterate, (Base.OneTo{Int64},))
    Base.Experimental.entrypoint(iterate, (Base.OneTo{Int64}, Int64))
    Base.Experimental.entrypoint(HiGHS.Highs_destroy, (J.Optimizer,))
    Base.Experimental.entrypoint(HiGHS.HighsCallbackDataIn, (Cint,))
end
