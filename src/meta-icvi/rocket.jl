__precompile__()

"""
Main module for the `Rocket.jl` method.

$(DOCSTRING_ATTRIBUTION)

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
module Rocket

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Full usings (which supports comma-separated import notation)
using
    DocStringExtensions,
    Random,
    NumericalTypeAliases

# Colon syntax broken into new lines
using JLD2: save_object, load_object
using StatsBase: sample
using PrecompileSignatures: @precompile_signatures  # Precompile concrete type methods

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Common docstring: description of attribution for the Rocket module for inclusion in relevant docstrings.
"""
DOCSTRING_ATTRIBUTION = """
# Attribution

## Programmer

- Sasha Petrenko <petrenkos@mst.edu> @AP6YC

## Original Authors

- Angus Dempster
- Francois Petitjean
- Geoff Webb

## Bibtex Entry

```bibtex
@article{dempster_etal_2020,
    author  = {Dempster, Angus and Petitjean, Francois and Webb, Geoffrey I},
    title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
    year    = {2020},
    journal = {Data Mining and Knowledge Discovery},
    doi     = {https://doi.org/10.1007/s10618-020-00701-z}
}
```

## Citation Links

- [preprint](https://arxiv.org/abs/1910.13051)
- [DOI](https://doi.org/10.1007/s10618-020-00701-z)
"""

# -----------------------------------------------------------------------------
# STRUCTURES
# -----------------------------------------------------------------------------

"""
Structure containing information about one rocket kernel.
"""
struct RocketKernel
    """
    The length of the features.
    """
    length::Int

    """
    The vector of weights corresponding to the features.
    """
    weight::Vector{Float}

    """
    The internal Rocket bias parameter, computed during construction.
    """
    bias::Float

    """
    The internal Rocket dilation parameter, computed during construction.
    """
    dilation::Int

    """
    The internal Rocket padding parameter, computed during construction.
    """
    padding::Int
end

"""
Structure containing a vector of rocket kernels.

# Attribution

## Programmer

- Sasha Petrenko <petrenkos@mst.edu> @AP6YC

## Original Authors

- Angus Dempster
- Francois Petitjean
- Geoff Webb

## Bibtex Entry

```bibtex
@article{dempster_etal_2020,
    author  = {Dempster, Angus and Petitjean, Francois and Webb, Geoffrey I},
    title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
    year    = {2020},
    journal = {Data Mining and Knowledge Discovery},
    doi     = {https://doi.org/10.1007/s10618-020-00701-z}
}
```

## Arxiv Preprint Link

https://arxiv.org/abs/1910.13051 (preprint)
"""
mutable struct RocketModule
    """
    The list of Rocket kernels constituting a full Rocket module.
    """
    kernels::Vector{RocketKernel}
end

"""
Create a new RocketModule structure, requiring feature length and the number of kernels.

# Arguments
- `input_length::Integer`: the desired length of the kernel features.
- `n_kernels::Integer`: the desired number of kernels to generate.
"""
function RocketModule(input_length::Integer, n_kernels::Integer)
    # Declare our candidate kernel lengths
    candidate_lengths = [7, 9, 11]

    # Instantiate the list of kernels
    kernels = Vector{RocketKernel}()

    # Iteratively create the kernels
    for _ = 1:n_kernels
        # Compute kernel parameters
        _length = sample(candidate_lengths)
        _weight = randn(_length)
        _bias = rand() * 2 - 1
        _dilation = Integer(floor(rand() * log2((input_length - 1) / (_length - 1))))
        _padding = Bool(rand(0:1)) ? Integer(floor(((_length - 1) * _dilation) / 2)) : 0
        # Create the kernel
        _kernel = RocketKernel(
            _length,
            _weight,
            _bias,
            _dilation,
            _padding
        )
        push!(kernels, _kernel)
    end

    RocketModule(kernels)
end

"""
Empty constructor for the RocketModule object.
"""
function RocketModule()
    # Create a default
    return RocketModule(5, 100)
end

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

"""
Apply a single RocketModule kernel to the sequence x.

# Arguments
- `kernel::RocketKernel`: rocket kernel used for computing features.
- `x::RealVector`: data sequence for computing rocket features.
"""
function apply_kernel(kernel::RocketKernel, x::RealVector)
    input_length = length(x)
    output_length = (input_length + (2 * kernel.padding)) - ((kernel.length - 1) * kernel.dilation)
    _ppv = 0
    _max = -Inf
    ending = (input_length + kernel.padding) - ((kernel.length - 1) * kernel.dilation)
    for i = -kernel.padding:ending
        _sum = kernel.bias
        index = i
        for j = 1:kernel.length
            if index > 0 && (index < input_length)
                _sum += kernel.weight[j] * x[index]
            end
            index += kernel.dilation
        end
        _max = max(_sum, _max)
        if _sum > 0
            _ppv += 1
        end
    end
    return [_ppv / output_length, _max]
end

"""
Run a vector of rocket kernels along a sequence x.

# Arguments
- `rocket::RocketModule`: rocket module containing many kernels for processing.
- `x::RealVector`: data sequence for computing rocket features.
"""
function apply_kernels(rocket::RocketModule, x::RealVector)
    # Get the number of kernels for preallocation and iteration
    n_kernels = length(rocket.kernels)

    # Preallocate the return array
    features = zeros(n_kernels, 2)

    # Calculate the features for each kernel
    for i = 1:n_kernels
        features[i, :] = apply_kernel(rocket.kernels[i], x)
    end

    # Return the full features array
    return features
end

"""
Save the rocket parameters to a .jld2 file.

# Arguments
`rocket::RocketModule`: rocket module to save.
`filepath::String`: path to .jld2 for saving rocket parameters. Defaults to rocket.jld2.
"""
function save_rocket(rocket::RocketModule, filepath::String="rocket.jld2")
    # Use the JLD2 save_object for simplicity
    save_object(filepath, rocket)
end

"""
Load and return a rocket module with existing parameters from a .jld2 file.

# Arguments
`filepath::String`: path to .jld2 containing rocket parameters. Defaults to rocket.jld2.
"""
function load_rocket(filepath::String="rocket.jld2")
    # Use the JLD2 load_object for simplicity
    return load_object(filepath)
end

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

# Export relevant names
export

    # Structs
    RocketKernel,
    RocketModule,

    # Methods
    apply_kernel,
    apply_kernels,
    load_rocket,
    save_rocket

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
@precompile_signatures(Rocket)

end
