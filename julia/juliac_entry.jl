# Register roots in the image-build session: registrations in a package's
# precompile cache are not retained by JuliaC.
using JuMPyHiGHS
import GenOpt
import HiGHS
import JuMP
import MathOptInterface

include("trim_roots.jl")
include("trim_dispatch.jl")

# Do not execute workload.jl here. JuliaC image generation does not initialize
# JLL libraries, so constructing a solver would fail before exercising the
# dynamic calls. trim_dispatch.jl is collected in an ordinary Julia process.
