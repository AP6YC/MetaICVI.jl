__precompile__()

module Rocket

# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# using StatsBase: ZScoreTransform, fit!
using
    Random,
    NumericalTypeAliases

using JLD2: save_object, load_object
using StatsBase: sample

# -----------------------------------------------------------------------------
# STRUCTURES
# -----------------------------------------------------------------------------

"""
    RocketKernel

Structure containing information about one rocket kernel.
"""
struct RocketKernel
    length::Int
    weight::Vector{Float}
    bias::Float
    dilation::Int
    padding::Int
end

"""
    RocketModule

Structure containing a vector of rocket kernels.

# References

## Authors
Angus Dempster, Francois Petitjean, Geoff Webb

## Bibtex Entry
@article{dempster_etal_2020,
  author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
  title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
  year    = {2020},
  journal = {Data Mining and Knowledge Discovery},
  doi     = {https://doi.org/10.1007/s10618-020-00701-z}
}

## Arxiv Preprint Link
https://arxiv.org/abs/1910.13051 (preprint)
"""
mutable struct RocketModule
    kernels::Vector{RocketKernel}
end

"""
    RocketModule(input_length::Integer, n_kernels::Integer)

Create a new RocketModule structure, requiring feature length and the number of kernels.
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
        _bias = rand()*2 - 1
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
    RocketModule()

Default constructor for the RocketModule object.
"""
function RocketModule()
    return RocketModule(5, 100)
end

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

"""
    apply_kernel(kernel::RocketKernel, x::RealVector)

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
    apply_kernels(rocket::RocketModule, x::RealVector)

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
    save_rocket(rocket::RocketModule, filepath::String="rocket.jld2")

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
    load_rocket(filepath::String="rocket.jld2")

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

end
