"""
    make.jl

This file builds the documentation for the ClusterValidityIndices.jl package
using Documenter.jl and other tools.
"""

# --------------------------------------------------------------------------- #
# DEPENDENCIES
# --------------------------------------------------------------------------- #

# using MetaICVI
using
    Documenter


# --------------------------------------------------------------------------- #
# SETUP
# --------------------------------------------------------------------------- #

# Fix GR headless errors
ENV["GKSwstype"] = "100"

# Get the current workind directory's base name
current_dir = basename(pwd())
@info "Current directory is $(current_dir)"

# If using the CI method `julia --project=docs/ docs/make.jl`
#   or `julia --startup-file=no --project=docs/ docs/make.jl`
if occursin("MetaICVI", current_dir)
    push!(LOAD_PATH, "../src/")
# Otherwise, we are already in the docs project and need to dev the above package
elseif occursin("docs", current_dir)
    Pkg.develop(path="..")
# Otherwise, building docs from the wrong path
else
    error("Unrecognized docs setup path")
end

# Inlude the local package
using MetaICVI

# using JSON
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

# --------------------------------------------------------------------------- #
# GENERATE
# --------------------------------------------------------------------------- #

DocMeta.setdocmeta!(MetaICVI, :DocTestSetup, :(using MetaICVI); recursive=true)

makedocs(;
    modules=[MetaICVI],
    authors="Sasha Petrenko",
    repo="https://github.com/AP6YC/MetaICVI.jl/blob/{commit}{path}#{line}",
    sitename="MetaICVI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://AP6YC.github.io/MetaICVI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
            "Guide" => "man/guide.md",
            "Examples" => "man/examples.md",
            "Contributing" => "man/contributing.md",
            "Index" => "man/full-index.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/AP6YC/MetaICVI.jl",
    devbranch = "main",
)
