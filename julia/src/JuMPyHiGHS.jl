module JuMPyHiGHS

import HiGHS

const Optimizer = HiGHS.Optimizer
_new_optimizer() = Optimizer()

include("entrypoints.jl")

end # module JuMPyHiGHS
