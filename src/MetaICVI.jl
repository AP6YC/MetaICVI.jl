__precompile__()

module MetaICVI

# Load the scikitlearn definitions here for runtime loading
using ScikitLearn: fit!, score, predict, @sk_import

# Load the PyObject definition at runtime because the compiled pointer will be different
function __init__()
    @eval @sk_import linear_model:RidgeClassifier
end

# Common types and method
include("common.jl")

# Meta-icvi module definition
include("meta-icvi/meta-icvi.jl")

export

    # Structures
    MetaICVIModule,
    MetaICVIOpts,

    # Methods
    get_metaicvi,
    train_and_save

end
