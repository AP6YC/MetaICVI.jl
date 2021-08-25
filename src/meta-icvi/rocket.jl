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


"""
struct RocketKernel
    length::Int64
    weight::Vector{Float64}
    bias::Float64
    dilation::Int64
    padding::Int64
end

mutable struct Rocket
    kernels::Vector{RocketKernel}
end

function Rocket(input_length::Int64, n_kernels::Int64)
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
        _dilation = Int64(floor(rand() * log2((input_length - 1) / (_length - 1))))
        _padding = Bool(rand(0:1)) ? Int64(floor(((_length - 1) * _dilation) / 2)) : 0
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

function apply_kernel(kernel::RocketKernel, x::Array)
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

function apply_kernels(rocket::Rocket, x::Vector)
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

# mutable struct Rocket
#     weights::Vector{Vector{Float64}}
#     lengths::Vector{Int64}
#     biases::Vector{Float64}
#     dilations::Vector{Int64}
#     paddings::Vector{Int64}
# end

# # function generate_kernels(input_length::Int64, num_kernels::Int64)
# function Rocket(input_length::Int64, num_kernels::Int64)
#     candidate_lengths = [7, 9, 11]
#     lengths = sample(candidate_lengths, num_kernels)

#     # weights = zeros(Float64, sum(lengths))
#     weights = Vector{Vector{Float64}}()
#     for i = 1:length(lengths)
#         push!(weights, zeros(Float64, lengths[i]))
#     end
#     biases = zeros(Float64, num_kernels)
#     dilations = zeros(Int64, num_kernels)
#     paddings = zeros(Int64, num_kernels)

#     for i = 1:num_kernels
#         _length = lengths[i]
#         weights[i] = randn(_length)
#         biases[i] = rand()*2 - 1
#         _dilation = Int64(floor(rand() * log2((input_length - 1) / (_length - 1))))
#         dilations[i] = _dilation
#         paddings[i] = Bool(rand(0:1)) ? Int64(floor(((_length - 1) * _dilation) / 2)) : 0
#     end

#     Rocket(
#         weights,
#         lengths,
#         biases,
#         dilations,
#         paddings
#     )
# end

# function apply_kernel(rocket::Rocket, x::Array, i::Int64)
#     input_length = length(x)
#     output_length = (input_length + (2 * rocket.paddings[i])) - ((rocket.lengths[i] - 1) * dilation)
#     _ppv = 0
#     _max = -Inf
#     ending = (input_length + rocket.paddings[i]) - ((rocket.lengths[i] - 1) * rocket.dilations[i])
#     for i = -rocket.paddings[i]:ending
#         _sum = rocket.biases[i]
#     end
# end
