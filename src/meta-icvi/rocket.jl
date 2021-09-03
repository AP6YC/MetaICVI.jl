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

using StatsBase
using DelimitedFiles
using Random

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------
#   **Taken from StatsBase.jl**
#
#  These types signficantly reduces the need of using
#  type parameters in functions (which are often just
#  for the purpose of restricting the arrays to real)
#
# These could be removed when the Base supports
# covariant type notation, i.e. AbstractVector{<:Real}

# Real-numbered aliases
const RealArray{T<:Real, N} = AbstractArray{T, N}
const RealVector{T<:Real} = AbstractArray{T, 1}
const RealMatrix{T<:Real} = AbstractArray{T, 2}

# Integered aliases
const IntegerArray{T<:Integer, N} = AbstractArray{T, N}
const IntegerVector{T<:Integer} = AbstractArray{T, 1}
const IntegerMatrix{T<:Integer} = AbstractArray{T, 2}

# Specifically floating-point aliases
const RealFP = Union{Float32, Float64}

# -----------------------------------------------------------------------------
# STRUCTURES
# -----------------------------------------------------------------------------

"""
    RocketKernel

Structure containing information about one rocket kernel.
"""
struct RocketKernel
    length::Integer
    weight::RealVector
    bias::RealFP
    dilation::Integer
    padding::Integer
end

"""
    RocketModule

Structure containing a vector of rocket kernels.
"""
mutable struct RocketModule
    kernels::Vector{RocketKernel}
end

"""
    RocketModule()

Default constructor for the RocketModule object.
"""
function RocketModule()
    return RocketModule(5, 100)
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
end # apply_kernel(kernel::RocketKernel, x::RealVector)

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
end # apply_kernels(rocket::RocketModule, x::RealVector)

"""
    save_rocket(rocket::RocketModule; filepath::String="rocket.csv")

Save the rocket parameters to a .csv file.

# Arguments
`rocket::RocketModule`: rocket module to save.
`filepath::String`: path to .csv for saving rocket parameters. Defaults to rocket.csv.
"""
function save_rocket(rocket::RocketModule; filepath::String="rocket.csv")

end # save_rocket(rocket::RocketModule; filepath::String="rocket.csv")

"""
    load_rocket(;filepath::String="rocket.csv")

Load and return a rocket module with existing parameters from a .csv file.

# Arguments
`filepath::String`: path to .csv containing rocket parameters. Defaults to rocket.csv.
"""
function load_rocket(;filepath::String="rocket.csv")

end # load_rocket(;filepath::String="rocket.csv")

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
