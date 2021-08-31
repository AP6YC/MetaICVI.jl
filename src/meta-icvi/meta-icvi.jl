using ClusterValidityIndices
using Logging
using Parameters

# Get the rocket kernel definitions
include("rocket.jl")

using .Rocket

"""
    MetaICVIOpts()

Meta-ICVI module options

# Examples
```julia-repl
julia> MetaICVIOpts()
```
"""
@with_kw mutable struct MetaICVIOpts @deftype Integer
    # Size of ICVI window: [1, infty]
    icvi_window = 5; @assert icvi_window >= 1
    # Size of correlation window: [1, infty)
    correlation_window = 5; @assert correlation_window >= 1
    # Number of rocket kernels: [1, infty)
    n_rocket = 5; @assert n_rocket >= 1
    # Display flag
    display::Bool = true
end # MetaICVIOpts

"""
    MetaICVI
"""
mutable struct MetaICVIModule
    opts::MetaICVIOpts
    cvis::Vector{AbstractCVI}
    criterion_values::RealVector
    correlations::RealVector
    rocket::RocketModule
end
