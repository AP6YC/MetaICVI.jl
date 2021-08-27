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
using Random

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
    Rocket

Structure containing a vector of rocket kernels
"""
mutable struct Rocket
    kernels::Vector{RocketKernel}
end

"""
    Rocket()

Default constructor for the Rocket object.
"""
function Rocket()
    return Rocket(5, 100)
end

"""
    Rocket(input_length::Integer, n_kernels::Integer)

Constructor for the Rocket structure, requiring feature length and the number of kernels.
"""
function Rocket(input_length::Integer, n_kernels::Integer)
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

    Rocket(kernels)
end

"""
    apply_kernel(kernel::RocketKernel, x::RealVector)

Apply a single Rocket kernel to the sequence x.
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
    apply_kernels(rocket::Rocket, x::RealVector)

Run a vector of rocket kernels along a sequence x.
"""
function apply_kernels(rocket::Rocket, x::RealVector)
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
end # apply_kernels(rocket::Rocket, x::RealVector)
