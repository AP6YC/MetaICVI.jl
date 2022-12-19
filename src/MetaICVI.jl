__precompile__()

"""
Main module for `MetaICVI.jl`, a Julia package implementing the MetaICVI method.

This module exports all of the modules, options, and utilities used by the `MetaICVI.jl package.`
For full usage, see the official guide at https://ap6yc.github.io/MetaICVI.jl/dev/man/guide/.

# Basic Usage

Install and import the package in a script with

```julia
using Pkg
Pkg.add("MetaICVI")
using MetaICVI
```

Next, create a MetaICVI module with some options

```julia
# Create the options
opts = MetaICVIOpts(
    fail_on_missing = false
)
# Create a module
metaicvi = MetaICVIModule(opts)
```

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
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
    DocStringExtensions,
    # using BSON
    # Custom libraries
    ClusterValidityIndices,         # All Julia-implemented CVI definitions
    NumericalTypeAliases            # Abstract type aliases

# Colon syntax broken into new lines
using StatsBase: corspearman        # Rank correlation for cvi criterion values
using ProgressMeter: @showprogress  # Data loading progress for training
using DelimitedFiles: readdlm       # Loading cvi data
using PrecompileSignatures: @precompile_signatures  # Precompile concrete type methods
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

# Common code
include("meta-icvi/common.jl")

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

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
@precompile_signatures(MetaICVI)

end
