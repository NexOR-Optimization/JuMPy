module JuMPyPaper
import MathOptInterface as MOI

include("storage.jl")
if get(ENV, "JUMPY_PAPER_BACKEND", "storage") == "gurobi"
    import Gurobi
    const Optimizer = Gurobi.Optimizer
elseif get(ENV, "JUMPY_PAPER_BACKEND", "storage") == "highs"
    import HiGHS
    const Optimizer = HiGHS.Optimizer
else
    const Optimizer = PaperStorage.Model{Float64}
end
include("../../julia/src/entrypoints.jl")

# Untimed diagnostics: export a small model for coefficient-level validation.
Base.@ccallable function jumpy_paper_write_model(model::Ptr{Cvoid}, path::Cstring)::Cint
    @_catch Cint(-1) begin
        snapshot = MOI.FileFormats.MOF.Model()
        MOI.copy_to(snapshot, JuMP.backend(_get(model).model))
        MOI.write_to_file(snapshot, unsafe_string(path))
        Cint(0)
    end
end
Base.@ccallable function jumpy_paper_num_variables(model::Ptr{Cvoid})::Int64
    @_catch Int64(-1) MOI.get(JuMP.backend(_get(model).model), MOI.NumberOfVariables())
end
Base.@ccallable function jumpy_paper_gc()::Cvoid
    GC.gc(true)
    return
end
end
