# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# External libraries

using Logging
using Parameters
using StatsBase: corspearman
using JLD
using PyCallJLD
using PyCall
using ScikitLearn
# using ScikitLearn: fit!, score, predict, @sk_import
# # @sk_import linear_model:RidgeClassifier
# @sk_import linear_model:RidgeClassifier
# classifier = RidgeClassifier()

# Custom libraries

using ClusterValidityIndices

# Local libraries

# Get the rocket kernel definitions
include("rocket.jl")
using .Rocket

# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# STRUCTURES
# -----------------------------------------------------------------------------

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
    rocket_file::String = "data/models/rocket.jld2"
    # Classifier file location
    classifier_file::String="data/models/classifier.jld"
    # Display flag
    display::Bool = true
    # Flag to fail if any file is missing (rather than creating new objects)
    fail_on_missing::Bool = false
end # MetaICVIOpts

"""
    MetaICVI

# Fields
- `opts::MetaICVIOpts`: options for construction.
- `cvis::Vector{AbstractCVI}`: list of cvis used for computing the CVIs.
- `criterion_values::RealVector`: list of outputs of the cvis used for computing correlations.
- `correlations::RealVector`: list of outputs of the rank correlations.
- `features::RealVector`: list of outputs of the rocket feature kernels.
- `rocket::RocketModule`: time-series random feature kernels module.
- `classifier::PyCall.PyObject`: ScikitLearn classifier.
- `performance::RealFP`: final output of the most recent the Meta-ICVI step.
"""
mutable struct MetaICVIModule
    opts::MetaICVIOpts
    cvis::Vector{AbstractCVI}
    criterion_values::Vector{RealVector}
    correlations::RealVector
    features::RealVector
    rocket::RocketModule
    classifier::PyCall.PyObject
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

    # Initialize the empty vectors for each criterion value
    cvi_values = [Array{RealFP}(undef, 0) for _ = 1:length(cvis)]

    # Construct the rocket kernels
    if isfile(opts.rocket_file)
        # If we have a file, load the module
        @info "Loading rocket kernels: $(opts.rocket_file)"
        rocket_module = load_rocket(opts.rocket_file)
    else
        # Otherwise, construct a module
        rocket_module = RocketModule(opts.correlation_window, opts.n_rocket)
        @warn "Missing/incorrect path for the rocket module file. \n\t Constructing a new rocket module"
        # If we specified a file but none was there, then save to that file
        if !isempty(opts.rocket_file)
            save_rocket(rocket_module, opts.rocket_file)
        else
            error("No filename provided for rocket file saving/loading.")
        end
    end

    # Load the classifier
    if isfile(opts.classifier_file)
        @info "Loading classifier: $(opts.classifier_file)"
        # classifier = JLD.load(opts.classifier_file, "learner")
        classifier = load_classifier(opts.classifier_file)
    # If we didn't get a valid file, check for errors or construct a new object
    else
        error_msg = "Missing/incorrect path for pretrained classifier file."
        # If we want to fail on a missing file, throw an error
        if opts.fail_on_missing
            error(error_msg)
        # Otherwise, put up a warning and create a new classifier.
        else
            # If a file is even provided, construct a classifier
            if !isempty(opts.classifier_file)
                @warn error_msg
                @info "Constructing a new classifier."
                classifier = RidgeClassifier()
                save_classifier(classifier, opts.classifier_file)
            # Otherwise, throw an error that no filename is provided
            else
                error("No filename provided for classifier saving/loading.")
            end
        end
    end

    # Construct and return the module
    return MetaICVIModule(
        opts,                       # opts
        cvis,                       # cvis
        cvi_values,                 # criterion_values
        Array{RealFP}(undef, 0),    # correlations
        Array{RealFP}(undef, 0),    # features
        rocket_module,              # rocket
        classifier,                 # classifier
        0.0                         # performance
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

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

"""
    load_classifier(filepath::String)

Load the classifier at the filepath.

# Arguments
`filepath::String`: location of the classifier .jld file.
"""
function load_classifier(filepath::String)
    return JLD.load(filepath, "classifier")
end # load_classifier(filepath::String)

"""
    save_classifier(classifier::PyCall.PyObject, filepath::String)

Save the classifier at the filepath.

# Arguments
`classifier::PyCall.PyObject`: classifier object to save.
`filepath::String`: name/path to save the classifier .jld file.
"""
function save_classifier(classifier::PyCall.PyObject, filepath::String)
    JLD.save(filepath, "classifier", classifier)
end # save_classifier(classifier::PyCall.PyObject, filepath::String)

"""
    train_and_save(metaicvi::MetaICVIModule, x::RealMatrix, y::IntegerVector)

Train the classifier on x/y and save the kernels and classifier.

# Arguments
`metaicvi::MetaICVIModule`: metaicvi module to save with.
`x::RealMatrix`: features to train on.
`y::IntegerVector`: correct/over/under partition targets.
"""
function train_and_save(metaicvi::MetaICVIModule, x::RealMatrix, y::IntegerVector)
    # Create a new classifier
    # TODO: option to train/retrain loaded/serialized models
    metaicvi.classifier = RidgeClassifier()
    # Train the classifier
    fit!(metaicvi.classifier, x, y)
    # Save the rocket kernels used
    save_rocket(metaicvi.rocket, metaicvi.opts.rocket_file)
    # Save the classifier used
    save_classifier(metaicvi.classifier, metaicvi.opts.classifier_file)
end

"""
    get_icvis(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

Compute and store the icvi criterion values.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
- `sample::RealVector`: the sample used for clustering.
- `label::Integer`: the label prescribed to the sample by the clustering algorithm.
"""
function get_icvis(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
    # Update all of the cvis incrementally
    for ix = 1:length(metaicvi.cvis)
        # Compute and push the criterion value
        value = get_icvi!(metaicvi.cvis[ix], sample, label)
        push!(metaicvi.criterion_values[ix], value)
        # FIFO the list to size
        while length(metaicvi.criterion_values[ix]) > metaicvi.opts.icvi_window
            popfirst!(metaicvi.criterion_values[ix])
        end
    end
end # get_icvis(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

"""
    get_correlations(metaicvi::MetaICVIModule)

Compute and store the rank correlations from the cvi values.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
"""
function get_correlations(metaicvi::MetaICVIModule)
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
end # get_correlations(metaicvi::MetaICVIModule)

"""
    get_rocket_features(metaicvi::MetaICVIModule)

Compute and store the rocket features.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
"""
function get_rocket_features(metaicvi::MetaICVIModule)
    # If there are enough correlations, compute compute the meta-icvi value
    if length(metaicvi.correlations) >= metaicvi.opts.correlation_window
        metaicvi.features = apply_kernels(metaicvi.rocket, metaicvi.correlations)[:, 1]
        # TODO
        metaicvi.performance = 0.0
    else
        metaicvi.features = zeros(metaicvi.opts.n_rocket)
    end
end # get_rocket_features(metaicvi::MetaICVIModule)

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
        # Compute the icvi values
        get_icvis(metaicvi, sample, label)

        # Compute the rank correlations
        get_correlations(metaicvi)

        # Compute the rocket features
        get_rocket_features(metaicvi)
    else
        # Default to 0
        metaicvi.performance = 0.0
    end

    return metaicvi.performance
end # get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
