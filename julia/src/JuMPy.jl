"""
    JuMPy

The Julia side of JuMPy's compiled backends. A backend is a package whose
module calls [`@entrypoints`](@ref) with the optimizer of its solver; JuliaC
compiles it into a library exposing the C ABI that the `jumpy` Python package
loads.
"""
module JuMPy

import GenOpt
import JuMP
import MathOptInterface as MOI

# Shared with the JuliaCall interface, which includes this file on its own.
include("JuMPyModel.jl")
include("JuMPyDirect.jl")
include("registry.jl")
include("native_sets.jl")
include("trim_roots.jl")
include("Client.jl")
include("TrimDispatch.jl")

"""
    ABI_VERSION

The version of the C ABI, returned by `jumpy_abi_version`. The `jumpy` Python
package only loads libraries built for the version it expects.
"""
const ABI_VERSION = 1

const _ENTRYPOINTS = joinpath(@__DIR__, "entrypoints.jl")

"""
    @entrypoints(optimizer, sets = JuMPy.supported_sets(optimizer))

Define the C entry points of a JuMPy backend for `optimizer`, the type of a
MathOptInterface optimizer constructible without arguments, in the calling
module. Python constructs the set types `sets`, a tuple of types; by default,
those that `optimizer` declares in its `MOI.supports_*` methods.

The module can then be compiled with JuliaC; see `julia/README.md`.

## Example

```julia
module JuMPyHiGHS
import HiGHS, JuMPy
JuMPy.@entrypoints HiGHS.Optimizer
end
```
"""
macro entrypoints(optimizer, sets = nothing)
    if sets === nothing
        sets = :($JuMPy.supported_sets(Optimizer))
    end
    return esc(quote
        import JuMPy
        using JuMPy: GenOpt, JuMP, MOI
        const Optimizer = $optimizer
        const SET_TYPES = $sets
        include($_ENTRYPOINTS)
    end)
end

end # module JuMPy
