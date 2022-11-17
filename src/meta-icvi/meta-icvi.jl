# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Local libraries
# Get the rocket kernel definitions
include("rocket.jl")
using .Rocket

using Random
using JLD2

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# const MetaICVIClassifier = ScikitLearn.Skcore.FitBit
const MetaICVIClassifier = PyCall.PyObject

# Top of the module for default paths
const module_dir(paths...) = joinpath(dirname(pathof(@__MODULE__)), "..", paths...)

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
    """
    Which scikitlearn classifier to load.
    """
    classifier_selection::Symbol = :SGDClassifier

    """
    Scikitlearn classifier keyword arguments.
    """
    classifier_opts::NamedTuple = (loss="log", max_iter=30)

    """
    Size of ICVI window: [1, infty).
    """
    icvi_window = 5; @assert icvi_window >= 1

    """
    Size of correlation window: [1, infty).
    """
    correlation_window = 5; @assert correlation_window >= 1

    """
    Number of rocket kernels: [1, infty).
    """
    n_rocket = 5; @assert n_rocket >= 1

    """
    Rocket file location.
    """
    rocket_file::String = module_dir("data", "models", "rocket.jld2")

    """
    Classifier file location.
    """
    classifier_file::String = module_dir("data", "models", "classifier.jld")
    # classifier_file::String = module_dir("data", "models", "classifier.jld2")

    """
    Display flag.
    """
    display::Bool = true

    """
    Flag to fail if any file is missing (rather than creating new objects).
    """
    fail_on_missing::Bool = false
end

"""
    MetaICVIModule

Stateful information for a single MetaICVI module.

# Fields
- `opts::MetaICVIOpts`: options for construction.
- `cvis::Vector{CVI}`: list of cvis used for computing the CVIs.
- `criterion_values::RealVector`: list of outputs of the cvis used for computing correlations.
- `correlations::RealVector`: list of outputs of the rank correlations.
- `features::RealVector`: list of outputs of the rocket feature kernels.
- `rocket::RocketModule`: time-series random feature kernels module.
- `classifier::MetaICVIClassifier`: ScikitLearn classifier.
- `performance::RealFP`: final output of the most recent the Meta-ICVI step.
- `is_pretrained::Bool`: internal flag for if the classifier is trained and ready for inference.
"""
mutable struct MetaICVIModule
    """
    Options for construction.
    """
    opts::MetaICVIOpts

    """
    List of cvis used for computing the CVIs.
    """
    cvis::Vector{CVI}

    """
    List of outputs of the cvis used for computing correlations.
    """
    criterion_values::Vector{Vector{Float}}

    """
    List of outputs of the rank correlations.
    """
    correlations::Vector{Float}

    """
    List of outputs of the rocket feature kernels.
    """
    features::Vector{Float}

    """
    Time-series random feature kernels module.
    """
    rocket::RocketModule

    """
    ScikitLearn classifier.
    """
    classifier::MetaICVIClassifier

    """
    Final output of the most recent the Meta-ICVI step.
    """
    performance::Float

    """
    Final output of the most recent the Meta-ICVI step.
    """
    probabilities::Vector{Float}

    """
    Internal flag for if the classifier is trained and ready for inference.
    """
    is_pretrained::Bool
end

"""
    MetaICVIModule(opts::MetaICVIOpts)

Instantiate a MetaICVIModule with given options.

# Arguments
- `opts::MetaICVIOpts`: options struct for the MetaICVI object.
"""
function MetaICVIModule(opts::MetaICVIOpts)
    # Load the correct classifier from the opts
    # if !@isdefined SGDClassifier
    @eval begin
        if !@isdefined $(opts.classifier_selection)
            @sk_import linear_model: $(opts.classifier_selection)
        end
    end
    # @eval @sk_import linear_model: $(opts.classifier_selection)

    # Construct the CVIs
    cvis = [
        PS(),
        GD43(),
    ]

    # Initialize the empty vectors for each criterion value
    cvi_values = [Array{Float}(undef, 0) for _ = 1:length(cvis)]

    # Construct the rocket kernels
    if isfile(opts.rocket_file)
        # If we have a file, load the module
        @info "Loading rocket kernels: $(opts.rocket_file)"
        rocket_module = load_rocket(opts.rocket_file)
        # Correct the expected number of kernels if necessary
        if opts.n_rocket != length(rocket_module.kernels)
            @warn "Provided incorrect number of kernels with loaded file, correcting MetaICVIModule options."
            opts.n_rocket = length(rocket_module.kernels)
        end
    else
        # Otherwise, construct a module
        @warn "Missing/incorrect path for the rocket module file."
        @info "Constructing a new rocket module"
        rocket_module = RocketModule(opts.correlation_window, opts.n_rocket)
    end

    # Load the classifier
    if isfile(opts.classifier_file)
        @info "Loading classifier: $(opts.classifier_file)"
        # classifier = JLD.load(opts.classifier_file, "learner")
        classifier = load_classifier(opts.classifier_file)
        is_pretrained = true
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
                classifier = construct_classifier(opts)
                is_pretrained = false
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
        Array{Float}(undef, 0),     # correlations
        Array{Float}(undef, 0),     # features
        rocket_module,              # rocket
        classifier,                 # classifier
        0.0,                        # performance
        zeros(3),                   # probabilities
        is_pretrained               # is_pretrained
    )
end

"""
    MetaICVIModule()

Default constructor for the MetaICVIModule.
"""
function MetaICVIModule()
    # Create the default options
    opts = MetaICVIOpts()
    # Return the Meta-ICVI module constructed with the default options
    return MetaICVIModule(opts)
end

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

"""
    Base.show(io::IO, metaicvi::MetaICVIModule)

Display a metaicvi module to the command line.

# Arguments
- `io::IO`: default io stream.
- `metaicvi::MetaICVIModule`: metaicvi object about which to display info.
"""
function Base.show(io::IO, metaicvi::MetaICVIModule)
    compact = get(io, :compact, false)
    if compact
        # print(metaicvi.opts)
        print("MetaICVIModule")
    else
        # show(io, metaicvi.opts)
        pretrain_display_flag = MetaICVI.is_pretrained(metaicvi)
        opts = metaicvi.opts
        print(
            """
            MetaICVIModule:
            \tperformance: $(metaicvi.performance)
            \tpretrained: $pretrain_display_flag
            Options:
            \tclassifier: $(opts.classifier_selection)
            \tclassifier_opts: $(opts.classifier_opts)
            \ticvi_window: $(opts.icvi_window)
            \tcorrelation_window: $(opts.correlation_window)
            \tn_rocket: $(opts.n_rocket)
            \trocket_file: $(opts.rocket_file)
            \tclassifier_file: $(opts.classifier_file)
            Flags:
            \tdisplay: $(opts.display)
            \tfail_on_missing: $(opts.display)
            """
        )
    end
end

"""
    construct_classifier(opts::MetaICVIOpts)

Construct a new classifier for the MetaICVI module with metaprogramming.

# Arguments
- `opts::MetaICVIOpts`: options containing the classifier type and options for instantiation.
"""
function construct_classifier(opts::MetaICVIOpts)
    @info "Constructing a new classifier"
    @eval classifier = $(opts.classifier_selection)(;$(opts.classifier_opts)...)
    return MetaICVIClassifier(classifier)
end

"""
    safe_save_classifier(metaicvi::MetaICVIModule)

Error handle saving of the metaicvi classifier.

# Arguments
- `metaicvi::MetaICVIModule`: metaicvi module containing the classifier and path for saving.
"""
function safe_save_classifier(metaicvi::MetaICVIModule)
    # If we specified a file but none was there, then save to that file
    if !isempty(metaicvi.opts.classifier_file)
        @info "Saving trained classifier"
        save_classifier(metaicvi.classifier, metaicvi.opts.classifier_file)
    else
        error("No filename provided for classifier file saving/loading.")
    end
end

"""
    safe_save_rocket(metaicvi::MetaICVIModule)

Error handle saving of the metaicvi rocket kernels.

# Arguments
- `metaicvi::MetaICVIModule`: metaicvi module containing the classifier and path for saving.
"""
function safe_save_rocket(metaicvi::MetaICVIModule)
    # If we specified a file but none was there, then save to that file
    if !isempty(metaicvi.opts.rocket_file)
        @info "Saving current rocket kernels"
        save_rocket(metaicvi.rocket, metaicvi.opts.rocket_file)
    else
        error("No filename provided for rocket file saving/loading.")
    end
end

"""
    load_classifier(filepath::String)

Load the classifier at the filepath.

# Arguments
- `filepath::String`: location of the classifier .jld file.
"""
function load_classifier(filepath::String)
    return MetaICVIClassifier(JLD.load(filepath, "classifier"))
    # return load_object(filepath)
    # return BSON.load(filepath, @__MODULE__)["classifier"]
end

"""
    save_classifier(classifier::MetaICVIClassifier, filepath::String)

Save the classifier at the filepath.

# Arguments
- `classifier::MetaICVIClassifier`: classifier object to save.
- `filepath::String`: name/path to save the classifier .jld file.
"""
function save_classifier(classifier::MetaICVIClassifier, filepath::String)
    JLD.save(filepath, "classifier", classifier)
    # save_object(filepath, classifier)
    # bson(filepath, Dict("classifier" => classifier))
end

"""
Saves the MetaICVI object, including its rocket kernels and serialized classifier.
"""
function save_metaicvi(metaicvi::MetaICVIModule)
    # Save the rocket kernels used
    safe_save_rocket(metaicvi)
    # Save the classifier used
    safe_save_classifier(metaicvi)
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
        value = get_cvi!(metaicvi.cvis[ix], sample, label)
        push!(metaicvi.criterion_values[ix], value)
        # FIFO the list to size
        while length(metaicvi.criterion_values[ix]) > metaicvi.opts.icvi_window
            popfirst!(metaicvi.criterion_values[ix])
        end
    end
end

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
end

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
    else
        metaicvi.features = zeros(metaicvi.opts.n_rocket)
    end

    # metaicvi.features = transpose([metaicvi.features])
end

"""
    is_pretrained(metaicvi::MetaICVIModule)

Checks if the classifier is pretrained to permit inference.

# Arguments
- `metaicvi::MetaICVIModule`: metaicvi module containing the classifier to check.
"""
function is_pretrained(metaicvi::MetaICVIModule)
    # Check if the model is pretrained with the isfit function
    # return ScikitLearn.Utils.isfit(metaicvi.classifier)
    return metaicvi.is_pretrained
end

"""
    get_probability(metaicvi::MetaICVIModule)

Compute and store the metaicvi value from the classifier.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
"""
function get_probability(metaicvi::MetaICVIModule)
    # If we have previously computed features
    if !isempty(metaicvi.features) && is_pretrained(metaicvi)
        # Compute the class 'probabilities'
        # probs = predict_proba(metaicvi.classifier, transpose([metaicvi.features]))
        # probs = predict_proba(metaicvi.classifier, metaicvi.features)
        probs = predict_proba(metaicvi.classifier, [metaicvi.features])
        metaicvi.probabilities = probs[:]

        # Store only the probability of correct partitioning
        metaicvi.performance = probs[1]
    else
        metaicvi.performance = 0.0
    end
end

"""
    get_features(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

Compute only the features on the sample and label without classifier inference.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
- `sample::RealVector`: the sample used for clustering.
- `label::Integer`: the label prescribed to the sample by the clustering algorithm.
"""
function get_features(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
    # Compute the icvi values
    get_icvis(metaicvi, sample, label)

    # Compute the rank correlations
    get_correlations(metaicvi)

    # Compute the rocket features
    get_rocket_features(metaicvi)
end

"""
    get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)

Compute and return the meta-icvi value.

# Arguments
- `metaicvi::MetaICVIModule`: the Meta-ICVI module.
- `sample::RealVector`: the sample used for clustering.
- `label::Integer`: the label prescribed to the sample by the clustering algorithm.
"""
function get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)
    # If the sample was not misclassified
    if label > 0
        # Compute the rocket features
        get_features(metaicvi, sample, label)

        # Compute and store the
        get_probability(metaicvi)
    else
        # Default to 0
        metaicvi.performance = 0.0
    end

    return metaicvi.performance
end

"""
    get_cvi_data(data_file::AbstractString)

Get the cvi data specified by the data_file path.

# Arguments
- `data_file::String`: file containing clustered data for cvi processing.
"""
function get_cvi_data(data_file::AbstractString)
    # Parse the data
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = convert(Matrix{Float}, data[1:2, :])
    train_y = convert(Vector{Int}, data[3, :])

    return train_x, train_y
end

"""
Loads the MetaICVI training data from the provided directory.
"""
function load_training_data(data_path::AbstractString)
    # Point to data
    data_dir(args...) = joinpath(data_path, args...)

    # Load the data
    correct_x, correct_y = get_cvi_data(data_dir("correct_partition.csv"))
    under_x, under_y = get_cvi_data(data_dir("under_partition.csv"))
    over_x, over_y = get_cvi_data(data_dir("over_partition.csv"))

    # Package the data conveniently
    data = Dict(
        "correct" => Dict(
            "x" => correct_x,
            "y" => correct_y
        ),
        "under" => Dict(
            "x" => under_x,
            "y" => under_y
        ),
        "over" => Dict(
            "x" => over_x,
            "y" => over_y
        )
    )

    return data
end

"""
Shuffles a batch of samples and their corresponding integer labels.
"""
function shuffle_x_y(x::RealMatrix, y::IntegerVector)
    n_samples = length(y)
    # Shuffle the data and targets
    ind_shuffle = Random.randperm(n_samples)
    temp_x = x[:, ind_shuffle]
    temp_y = y[ind_shuffle]
    return temp_x, temp_y
end

"""
Splits a batch of samples and their labels into train and test matrices.
"""
function split_x_y(x::RealMatrix, y::IntegerVector ; split::Real=0.8)
    n_samples = length(y)

    # Split into train/test
    split_ind = Int(floor(n_samples*split))

    train_x = x[:, 1:split_ind]
    test_x = x[:, split_ind+1:end]

    train_y = y[1:split_ind]
    test_y = y[split_ind+1:end]

    return train_x, test_x, train_y, test_y
end

"""
Splits the MetaICVI data dictionary into train and test data dictionaries.
"""
function split_training_data(data::Dict ; split::Real=0.8, shuffle=true)

    train_data, test_data = Dict(), Dict()

    for (key, value) in data

        train_data[key] = Dict()
        test_data[key] = Dict()

        # Shuffle
        if shuffle
            temp_x, temp_y = shuffle_x_y(value["x"], value["y"])
        else
            temp_x = value["x"]
            temp_y = value["y"]
        end

        (
            train_data[key]["x"],
            test_data[key]["x"],
            train_data[key]["y"],
            test_data[key]["y"]
        ) = (
            split_x_y(temp_x, temp_y, split=split)
        )
    end

    return train_data, test_data
end

"""
Takes the MetaICVI data dictionary and creates a matrix of samples and vector of labels for all partition qualities.
"""
function serialize_data(data::Dict)
    return_x = reduce(hcat, data[type]["x"] for (type, _) in data )
    return_y = reduce(vcat, data[type]["y"] for (type, _) in data )

    return return_x, return_y
end

"""
Takes the MetaICVI data dictionary and computes the features for training the classifier.
"""
function get_training_features(metaicvi::MetaICVIModule, data::Dict)

    # Create the target containers
    # data_lengths = [length(correct_y), length(under_y), length(over_y)]
    # offset_lengths = [0, length(correct_y), length(under_y)]
    data_lengths = [length(data["correct"]["y"]), length(data["under"]["y"]), length(data["over"]["y"])]
    offset_lengths = [0, length(data["correct"]["y"]), length(data["under"]["y"])]

    data_length = sum(data_lengths)
    features_data = zeros(metaicvi.opts.n_rocket, data_length)
    features_targets = zeros(Int, data_length)

    # Mapping of datatype to numeral target for classification
    type_to_num = Dict(
        "correct" => 1,
        "under" => 2,
        "over" => 3
    )

    # Itereate over all data to get features
    for (type, subdata) in data
        # Get the offset directly from the type mapping and explicit definition
        data_offset = offset_lengths[type_to_num[type]]

        # Iterate over all entries of the type of partitioning (i.e., correct, under, and over)
        @showprogress for i = 1:length(subdata["y"])
            # Extract the sample and label
            sample = subdata["x"][:, i]
            label = subdata["y"][i]

            # Compute and retrieve the features
            get_features(metaicvi, sample, label)
            features = metaicvi.features

            # Save the results
            features_data[:, i + data_offset] = features
            features_targets[i + data_offset] = type_to_num[type]
        end
    end

    # return transpose(features_data), features_targets
    return features_data, features_targets
end

"""
    train_and_save(metaicvi::MetaICVIModule, x::RealMatrix, y::IntegerVector)

Train the classifier on x/y and save the kernels and classifier.

# Arguments
- `metaicvi::MetaICVIModule`: metaicvi module to save with.
- `x::RealMatrix`: features to train on.
- `y::IntegerVector`: correct/over/under partition targets.
"""
function train_and_save(metaicvi::MetaICVIModule, x::RealMatrix, y::IntegerVector)
# function train_and_save(metaicvi::MetaICVIModule, data_path::AbstractString)
    # Create a new classifier
    # TODO: option to train/retrain loaded/serialized models
    classifier = construct_classifier(metaicvi.opts)
    metaicvi.classifier = classifier

    # Train the classifier
    @info "Training classifier"
    fit!(metaicvi.classifier, x', y)
    metaicvi.is_pretrained = true

    # Save the metaicvi kernels and classifier
    save_metaicvi(metaicvi)
end

# mutable struct MetaICVIData
#     correct::Dict{String, }
# end