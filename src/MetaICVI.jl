__precompile__()

module MetaICVI

# --------------------------------------------------------------------------- #
# DEPENDENCIES
# --------------------------------------------------------------------------- #

# Full usings (which supports comma-separated import notation)
using
    # External libraries
    Logging,                        # Logging is used for operation diagnostics
    Parameters,                     # MetaICVIOpts are Parameters structs
    PyCall,                         # PyCall object definition
    JLD,                            # JLD is currently recommended for saving/loading ScikitLearn objects
    PyCallJLD,                      # PyCall definition for serialization with JLD
    ScikitLearn,                    # Classifiers are scikit-learn pyobjects
    # using BSON
    # Custom libraries
    ClusterValidityIndices,         # All Julia-implemented CVI definitions
    NumericalTypeAliases            # Abstract type aliases

# Colon syntax broken into new lines
using StatsBase: corspearman        # Rank correlation for cvi criterion values
using ProgressMeter: @showprogress  # Data loading progress for training
using DelimitedFiles: readdlm       # Loading cvi data
# using ScikitLearn.Skcore: FitBit

# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------

# Load the scikitlearn definitions here for runtime loading
# using ScikitLearn: fit!, score, predict, predict_proba, @sk_import

# Load the PyObject definition at runtime because the compiled pointer will be different
# function __init__()
#     # @eval @sk_import linear_model:RidgeClassifier
#     @eval @sk_import linear_model:SGDClassifier
    # @eval using PyCallJLD
    # @eval @sk_import linear_model as lm
# end

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

# Meta-icvi module definition
include("meta-icvi/meta-icvi.jl")

# -----------------------------------------------------------------------------
# EXPORTED NAMES
# -----------------------------------------------------------------------------

export

    # Structures
    MetaICVIModule,
    MetaICVIOpts,

    # Methods
    get_metaicvi,           # Compute features and "performance"
    get_features,           # Convenience function for just features
    get_training_features,  # Load data and process features for classifier training
    train_and_save

end
