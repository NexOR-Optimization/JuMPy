# Load the NexOR native ABI and register roots in the image-build session.
include("src/JuMPyNexOR.jl")
import GenOpt
import JuMP
import MathOptInterface

if Base.JLOptions().trim != 0
    include("trim_roots.jl")
    include("nexor_dispatch.jl")
end
