"""
    make.jl

# Description
This file builds the documentation for the MetaICVI.jl package
using Documenter.jl and other tools.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# --------------------------------------------------------------------------- #
# DEPENDENCIES
# --------------------------------------------------------------------------- #

# using MetaICVI
using
    Documenter,
    DemoCards,
    Pkg

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

# DocMeta.setdocmeta!(MetaICVI, :DocTestSetup, :(using MetaICVI); recursive=true)

# Generate the demo files
# this is the relative path to docs/
demopage, postprocess_cb, demo_assets = makedemos("examples")

assets = [
    joinpath("assets", "favicon.ico"),
]

# if there are generated css assets, pass it to Documenter.HTML
isnothing(demo_assets) || (push!(assets, demo_assets))

# Make the documentation
makedocs(
    modules=[MetaICVI],
    authors="Sasha Petrenko",
    repo="https://github.com/AP6YC/MetaICVI.jl/blob/{commit}{path}#{line}",
    sitename="MetaICVI.jl",
    format=Documenter.HTML(;
    prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://AP6YC.github.io/MetaICVI.jl",
        assets = assets,
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
            "Guide" => "man/guide.md",
            demopage,
            "Contributing" => "man/contributing.md",
            "Index" => "man/full-index.md"
        ]
    ],
)

# 3. postprocess after makedocs
postprocess_cb()

# -----------------------------------------------------------------------------
# DEPLOY
# -----------------------------------------------------------------------------

deploydocs(;
    repo="github.com/AP6YC/MetaICVI.jl",
    devbranch = "devbranch",
)
