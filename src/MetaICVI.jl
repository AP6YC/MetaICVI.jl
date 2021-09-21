__precompile__()

module MetaICVI

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

# Common types and method
include("common.jl")

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
