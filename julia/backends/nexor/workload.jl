# Traced in ordinary Julia before building the image; see julia/README.md.
# The solve is answered by a mock transport instead of the server.
import JuMPy
import JuMPyNexOR
import NexOR

const J = JuMPyNexOR

function optimize(m)
    model = J._get(m).model
    client = J.JuMP.backend(model)
    primal = ones(J.MOI.get(client, J.MOI.NumberOfVariables()))
    function request(_, method, path; body = "")
        path == "/problems" && return Dict("problem_id" => "w", "status" => "queued")
        startswith(path, "/problems/w/wait") &&
            return Dict("problem_id" => "w", "status" => "finished")
        attributes = Dict(
            "termination_status" => "OPTIMAL",
            "primal_status" => "FEASIBLE_POINT",
            "dual_status" => "NO_SOLUTION",
            "result_count" => 1,
            "objective_value" => 1.0,
            "raw_status" => "workload",
        )
        return Dict("solution" => Dict("attributes" => attributes, "primal" => primal))
    end
    NexOR._optimize!(client, request)
    # The mock transport bypasses the JuMP optimize wrapper.
    model.is_model_dirty = false
    return J.jumpy_termination_status(m)
end

JuMPy.Client.workload(
    J;
    attributes = (
        "solver" => "HiGHS",
        "mip_detect_symmetry" => true,
        "random_seed" => 1,
        "mip_rel_gap" => 1e-4,
    ),
    optimize,
)
