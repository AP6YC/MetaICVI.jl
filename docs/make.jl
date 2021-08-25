using MetaICVI
using Documenter

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
    ],
)

deploydocs(;
    repo="github.com/AP6YC/MetaICVI.jl",
    devbranch = "main",
)
