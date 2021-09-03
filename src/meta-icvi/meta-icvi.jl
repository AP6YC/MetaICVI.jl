using ClusterValidityIndices
using Logging
using Parameters

# Get the rocket kernel definitions
include("rocket.jl")

using .Rocket

"""
    MetaICVIOpts()

Meta-ICVI module options.

# Examples
```julia-repl
julia> MetaICVIOpts()
```
"""
@with_kw mutable struct MetaICVIOpts @deftype Int
    # Size of ICVI window: [1, infty)
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

"""
    MetaICVIModule(opts::MetaICVIOpts)

Instantiate a MetaICVIModule with given options.

# Arguments
- `opts::MetaICVIOpts`: options struct for the MetaICVI object.
"""
function MetaICVIModule(opts::MetaICVIOpts)
    # Construct the CVIs
    cvis = [
        PS(),
        GD43()
    ]

    # Construct and return the module
    return MetaICVIModule(
        opts,
        cvis,
        Array{RealFP}(undef, 0),
        Array{RealFP}(undef, 0),
        RocketModule()
    )
end # MetaICVIModule(opts::MetaICVIOpts)

"""
    MetaICVIModule()

Default constructor for the MetaICVIModule.
"""
function MetaICVIModule()
    opts = MetaICVIOpts()
    return MetaICVIModule(opts)
end

function get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
    # If the sample was not misclassified and we have a big enough window
    if label != -1 && label > 0
        # Update all of the cvis incrementally
        for ix = 1:n_cvis
            value = get_icvi!(metrics.cvis[ix], sample, label)
            push!(metrics.values[ix], value)
        end
        # If the window is big enough, compute the performance
        if length(metrics.values[1]) >= metrics.n_window
            # Get the spearman correlation
            performance = corspearman(metrics.values[1], metrics.values[2])/2 + 0.5
            # Sanitize a potential NaN response
            metrics.performance = isequal(performance, NaN) ? 0 : performance
        end
    else
        # Default to 0
        metrics.performance = 0
    end
end

# # Create a container for their criterion values for quick access
#     # values = zeros(Float64, length(cvis))
#     values = Vector{Vector{Float64}}()
#     for i = 1:length(cvis)
#         push!(values, Vector{Float64}())
#     end

#     # Total performance
#     performance = 0

#     # Default number of windows
#     n_window = 5

#     # Construct the metrics container
#     TaskDetectorMetrics(
#         cvis,               # cvis
#         values,             # values
#         performance,        # performance
#         n_window            # n_window
#     )
# end # TaskDetectorMetrics()

# """
#     update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Int)

# Update the task detector's metrics using ICVIs.

# # Fields
# - `metrics::TaskDetectorMetrics`: the metrics object being updated.
# - `sample::Array`: the array of features that are clustered to the label.
# - `label::Int`: the label prescribed by the clustering algorithm.
# """
# function update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Integer)
#     # Get the number of cvis each time, accomodating changes during operation
#     n_cvis = length(metrics.cvis)

#     # If the sample was not misclassified and we have a big enough window
#     if label != -1
#         # Update all of the cvis incrementally
#         for ix = 1:n_cvis
#             value = get_icvi!(metrics.cvis[ix], sample, label)
#             push!(metrics.values[ix], value)
#         end
#         # If the window is big enough, compute the performance
#         if length(metrics.values[1]) >= metrics.n_window
#             # Get the spearman correlation
#             performance = corspearman(metrics.values[1], metrics.values[2])/2 + 0.5
#             # Sanitize a potential NaN response
#             metrics.performance = isequal(performance, NaN) ? 0 : performance
#         end
#     else
#         # Default to 0
#         metrics.performance = 0
#     end

#     # FIFO the list
#     for ix = 1:n_cvis
#         while length(metrics.values[ix]) > metrics.n_window
#             popfirst!(metrics.values[ix])
#         end
#     end

#     # Calculate the performance
#     # metrics.performance = sigmoid(8*(mean(metrics.values) - 0.5))

#     # Return that performance
#     return metrics.performance
# end # update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Integer)
