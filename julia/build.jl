# Build one image. Never load the two profiles in the same process.
# julia --project=@juliac build.jl shared|trimmed [output-directory] [project]
using JuliaC

function build(args)
    args = isempty(args) ? ["shared"] : args
    if !(args[1] in ("shared", "trimmed")) || length(args) > 3
        error("Usage: build.jl shared|trimmed [output-directory] [project]")
    end
    trimmed = args[1] == "trimmed"
    output = abspath(length(args) >= 2 ? args[2] : joinpath(@__DIR__, "build-" * args[1]))
    mkpath(output)
    mktempdir() do staging
        project = if length(args) == 3
            abspath(args[3])
        else
            # Keep the copied project small, including on subsequent builds.
            # JuliaC itself copies its project before instantiating it.
            for file in ("Project.toml", "Manifest.toml", "src")
                cp(joinpath(@__DIR__, file), joinpath(staging, file))
            end
            staging
        end
        c_source = joinpath(staging, "runtime_info.c")
        cp(joinpath(@__DIR__, "runtime_info.c"), c_source)
        image = JuliaC.ImageRecipe(
            output_type = "--output-lib",
            file = joinpath(@__DIR__, "juliac_entry.jl"),
            project = project,
            trim_mode = trimmed ? "unsafe-warn" : nothing,
            add_ccallables = true,
            c_sources = [c_source],
            cflags = ["-DJUMPY_TRIMMED=$(Int(trimmed))"],
            jl_options = Dict("handle-signals" => "no", "threads" => "1"),
        )
        link = JuliaC.LinkRecipe(
            image_recipe = image,
            outname = joinpath(output, "libjumpy_highs"),
            rpath = "@bundle",
        )
        bundle = JuliaC.BundleRecipe(link_recipe = link, output_dir = output)
        JuliaC.compile_products(image)
        JuliaC.link_products(link)
        JuliaC.bundle_products(bundle)
        println("Built ", args[1], " image: ", link.outname)
    end
end

build(ARGS)
