using JuMP
import GenOpt
import HiGHS
import NexOR
import MathOptInterface as MOI
import JSON
include("models.jl")
# Use the same variable-index handling as the ordinary NexOR backend.
if !isdefined(Main, :JuMPyNexOR)
    Base.include(Main, joinpath(@__DIR__, "../../julia/src/JuMPyNexOR.jl"))
end
include("grouped_models.jl")

function parse_args(args)
    return Dict(args[i] => args[i+1] for i = 1:2:length(args))
end

function main(args)
    options = parse_args(args)
    mode = get(options, "--mode", "direct")
    mode in ("cached", "direct", "genopt") || error("Unknown mode: $mode")
    backend = get(options, "--backend", "highs")
    backend in ("highs", "nexor") || error("Unknown backend: $backend")
    problem = options["--model"]
    backend == "highs" &&
        problem == "facility" &&
        error("Facility requires NexOR's local model storage")
    sizes = parse.(Int, split(options["--sizes"], ','))
    repeats = parse(Int, get(options, "--repeats", "3"))
    warmup = parse(Int, get(options, "--warmup", "5"))
    factory = backend == "highs" ? HiGHS.Optimizer : NexOR.Optimizer
    builder = getfield(mode == "genopt" ? GroupedPaperModels : PaperModels, Symbol(problem))
    function construct(n)
        model = mode == "cached" ? Model(factory) : concrete_direct_model(factory())
        set_string_names_on_creation(model, false)
        set_silent(model)
        if backend == "highs"
            set_attribute(model, "threads", 1)
            if problem != "latency"
                set_time_limit_sec(model, 0.0)
                set_attribute(model, "presolve", "off")
            end
        end
        builder(model, n)
        if backend == "highs"
            optimize!(model)
        elseif mode == "cached"
            MOI.Utilities.attach_optimizer(JuMP.backend(model))
        end
        return model
    end
    function export_model(model)
        GroupedPaperModels.JuMPyModel.write_mof(model, options["--export"])
        return
    end
    started = time_ns()
    if warmup > 0
        model = construct(warmup)
        haskey(options, "--export") && export_model(model)
        model = nothing
    end
    println(
        JSON.json((;
            event = "warmup",
            frontend = "jump-$mode",
            backend,
            model = problem,
            warmup_seconds = (time_ns() - started) / 1e9,
        )),
    )
    flush(stdout)
    for n in sizes, sample = 0:(repeats-1)
        started = time_ns()
        model = construct(n)
        seconds = (time_ns() - started) / 1e9
        export_seconds = if haskey(options, "--export")
            started = time_ns()
            export_model(model)
            (time_ns() - started) / 1e9
        else
            nothing
        end
        objective =
            problem == "latency" && backend == "highs" ? objective_value(model) : nothing
        objective === nothing || @assert isapprox(objective, 205.0; atol = 1e-7)
        println(
            JSON.json((;
                event = "sample",
                frontend = "jump-$mode",
                backend,
                model = problem,
                size = n,
                sample,
                seconds,
                export_seconds,
                variables = num_variables(model),
                peak_rss_bytes = Sys.maxrss(),
                julia = string(VERSION),
                jump = string(pkgversion(JuMP)),
                genopt = string(pkgversion(GenOpt)),
                solver = string(pkgversion(backend == "highs" ? HiGHS : NexOR)),
                objective,
            )),
        )
        flush(stdout)
        model = nothing
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
