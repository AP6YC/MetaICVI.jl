# Package Guide

The `MetaICVI.jl` package is built upon modules that contain all of the state information during training and inference.
The MetaICVI modules are driven by options, which are themselves mutable keyword argument structs from the [Parameters.jl](https://github.com/mauro3/Parameters.jl) package.

To work with `MetaICVI.jl`, you should know:

- [How to install the package](@ref installation)
- [MetaICVI module basics](@ref metaicvi_modules)
- [How to use MetaICVI module options](@ref metaicvi_options)

## [Installation](@id installation)

The MetaICVI.jl package can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```julia
pkg> add MetaICVI
```

Alternatively, it can be added to your environment in a script with

```julia
using Pkg
Pkg.add("MetaICVI")
```

If you wish to have the latest changes between releases, you can directly add the GitHub repo as a dependency with

```julia
pkg> add https://github.com/AP6YC/MetaICVI.jl
```

## [MetaICVI Modules](@id metaicvi_modules)

TODO

## [MetaICVI Options](@id metaicvi_options)

TODO
