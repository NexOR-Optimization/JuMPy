using JuMP
import MathOptInterface as MOI
import JSON
include("models.jl")
include("grouped_models.jl")
include("storage.jl")

function parse_args(args)
    values = Dict{String,String}()
    i = 1
    while i <= length(args)
        values[args[i]] = args[i+1]
        i += 2
    end
    return values
end

function export_model(model, path)
    snapshot = MOI.FileFormats.MOF.Model()
    MOI.copy_to(snapshot, JuMP.backend(model))
    MOI.write_to_file(snapshot, path)
    return
end

function main(args)
    options = parse_args(args)
    mode = get(options, "--mode", "direct")
    mode in ("cached", "direct", "genopt") || error("Unknown mode: $mode")
    backend = get(options, "--backend", "storage")
    problem = options["--model"]
    sizes = parse.(Int, split(options["--sizes"], ','))
    repeats = parse(Int, get(options, "--repeats", "3"))
    warmup = parse(Int, get(options, "--warmup", "5"))
    factory = if backend == "gurobi"
        Gurobi.Optimizer
    elseif backend == "highs"
        HiGHS.Optimizer
    else
        PaperStorage.Model{Float64}
    end
    builder = getfield(mode == "genopt" ? GroupedPaperModels : PaperModels, Symbol(problem))
    function construct(n)
        model = mode == "cached" ? Model(factory) : concrete_direct_model(factory())
        set_string_names_on_creation(model, false)
        set_silent(model)
        if backend != "storage"
            set_attribute(model, backend == "gurobi" ? "Threads" : "threads", 1)
        end
        if backend != "storage" && problem != "latency"
            set_time_limit_sec(model, 0.0)
            set_attribute(
                model,
                backend == "gurobi" ? "Presolve" : "presolve",
                backend == "gurobi" ? 0 : "off",
            )
        end
        builder(model, n)
        if backend != "storage"
            optimize!(model)
        elseif mode == "cached"
            MOI.Utilities.attach_optimizer(JuMP.backend(model))
        end
        return model
    end
    started = time_ns()
    warmup > 0 && construct(warmup)
    warmup_seconds = (time_ns() - started) / 1e9
    println(
        JSON.json((;
            event = "warmup",
            frontend = "jump-$mode",
            backend,
            model = problem,
            warmup_seconds,
        )),
    )
    flush(stdout)
    for n in sizes, sample = 0:(repeats-1)
        GC.gc(true)
        started = time_ns()
        model = construct(n)
        seconds = (time_ns() - started) / 1e9
        if haskey(options, "--export")
            # Validation must not compile the MOF exporter in latency runs.
            Base.invokelatest(export_model, model, options["--export"])
        end
        objective =
            problem == "latency" && backend != "storage" ? objective_value(model) : nothing
        if objective !== nothing
            @assert isapprox(objective, 205.0; atol = 1e-7)
        end
        println(
            JSON.json((;
                event = "sample",
                frontend = "jump-$mode",
                backend,
                model = problem,
                size = n,
                sample,
                seconds,
                variables = num_variables(model),
                peak_rss_bytes = Sys.maxrss(),
                julia = string(VERSION),
                jump = string(pkgversion(JuMP)),
                genopt = string(pkgversion(GroupedPaperModels.GenOpt)),
                solver = backend == "highs" ? string(pkgversion(HiGHS)) : backend,
                objective,
            )),
        )
        flush(stdout)
        model = nothing
    end
    return
end
if abspath(PROGRAM_FILE) == @__FILE__
    index = findfirst(==("--backend"), ARGS)
    if index !== nothing && ARGS[index+1] == "gurobi"
        @eval import Gurobi
    elseif index !== nothing && ARGS[index+1] == "highs"
        @eval import HiGHS
    end
    main(ARGS)
end
