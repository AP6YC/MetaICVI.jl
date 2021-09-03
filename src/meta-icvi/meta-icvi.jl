using ClusterValidityIndices
using Logging
using Parameters
using MLJ

# Get the rocket kernel definitions
include("rocket.jl")

using .Rocket
using StatsBase

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
    # Rocket file location
    rocket_file::String = ""
    # Display flag
    display::Bool = true
end # MetaICVIOpts

"""
    MetaICVI

# Fields
- `opts::MetaICVIOpts`: options for construction.
- `cvis::Vector{AbstractCVI}`: list of cvis used for computing the CVIs.
- `criterion_values::RealVector`: list of outputs of the cvis used for computing correlations.
- `correlations::RealVector`: list of outputs of the rank correlations.
- `rocket::RocketModule`: time-series random feature kernels module.
- `performance::RealFP`: final output of the most recent the Meta-ICVI step.
"""
mutable struct MetaICVIModule
    opts::MetaICVIOpts
    cvis::Vector{AbstractCVI}
    criterion_values::Vector{RealVector}
    correlations::RealVector
    rocket::RocketModule
    performance::RealFP
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

    cvi_values = [Array{RealFP}(undef, 0) for i=1:length(cvis)]

    # Construct the rocket kernels
    if isfile(opts.rocket_file)
        # If we have a file, load the module
        rocket_module = load_rocket(opts.rocket_file)
    else
        # Otherwise, construct a module
        rocket_module = RocketModule(opts.correlation_window, opts.n_rocket)
        # If we specified a file but none was there, then save to that file
        if !isempty(opts.rocket_file)
            save_rocket(rocket_module, opts.rocket_file)
        end
    end

    # Construct and return the module
    return MetaICVIModule(
        opts,
        cvis,
        cvi_values,
        Array{RealFP}(undef, 0),
        rocket_module,
        0.0
    )
end # MetaICVIModule(opts::MetaICVIOpts)

"""
    MetaICVIModule()

Default constructor for the MetaICVIModule.
"""
function MetaICVIModule()
    # Create the default options
    opts = MetaICVIOpts()
    # Return the Meta-ICVI module constructed with the default options
    return MetaICVIModule(opts)
end # MetaICVIModule()

"""
    get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

Compute and return the meta-icvi value.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
- `sample::RealVector`: the sample used for clustering.
- `label::Integer`: the label prescribed to the sample by the clustering algorithm.
"""
function get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
    # If the sample was not misclassified and we have a big enough window
    if label != -1 && label > 0
        # Update all of the cvis incrementally
        for ix = 1:length(metaicvi.cvis)
            # Compute and push the criterion value
            # @info sample
            # @info label
            value = get_icvi!(metaicvi.cvis[ix], sample, label)
            push!(metaicvi.criterion_values[ix], value)
            # FIFO the list to size
            while length(metaicvi.criterion_values[ix]) > metaicvi.opts.icvi_window
                popfirst!(metaicvi.criterion_values[ix])
            end
        end

        # If the cvi window is big enough, compute the correlations
        if length(metaicvi.criterion_values[1]) >= metaicvi.opts.icvi_window
            # Get the spearman correlation
            correlation = corspearman(metaicvi.criterion_values[1], metaicvi.criterion_values[2])
            # Sanitize a potential NaN response
            # metaicvi.performance = isequal(performance, NaN) ? 0 : performance
            push!(metaicvi.correlations, correlation)
            # FIFO the list to size
            while length(metaicvi.correlations) > metaicvi.opts.correlation_window
                popfirst!(metaicvi.correlations)
            end
        end

        # If there are enough correlations, compute compute the meta-icvi value
        if length(metaicvi.correlations) >= metaicvi.opts.correlation_window
            features = apply_kernels(metaicvi.rocket, metaicvi.correlations)
            # TODO
            metaicvi.performance = 0.0
        end
    else
        # Default to 0
        metaicvi.performance = 0.0
    end

    return metaicvi.performance
end # get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

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
