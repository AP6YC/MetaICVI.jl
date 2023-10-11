"""
    test_utils.jl

# Description
This file defines some common utilities for testing, such as loading and formatting datasets for testing.
"""
# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    DelimitedFiles,
    Logging,
    NumericalTypeAliases,
    Random,
    Statistics
# using StatsBase

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
A basic struct for encapsulating the four components of supervised training.
"""
mutable struct DataSplit
    train_x::Array
    test_x::Array
    train_y::Array
    test_y::Array
end

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

const ARG_DATA_X = """
- `data_x::RealMatrix`: the matrix of features.
"""

const ARG_DATA_Y = """
- `data_y::IntegerVector`: the integered class labels as a vector.
"""

const ARG_RATIO = """
- `ratio::Real`: the fractional ratio for the train/test split.
"""

const ARG_SHUFFLE = """
- `shuffle::Bool=false`: optional, flag to preshuffle the data set.
"""

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Return a DataSplit struct that is split by the ratio (e.g. 0.8).

# Arguments
$ARG_DATA_X
$ARG_DATA_Y
$ARG_RATIO
$ARG_SHUFFLE
"""
function DataSplit(data_x::RealMatrix, data_y::IntegerVector, ratio::Real ; shuffle::Bool=false)
    dim, n_samples = size(data_x)
    split_ind = Int(floor(n_samples*ratio))

    if shuffle
        # Shuffle the data and targets
        ind_shuffle = Random.randperm(n_samples)
        temp_data_x = data_x[:, ind_shuffle]
        temp_data_y = data_y[ind_shuffle]
    else
        temp_data_x = data_x
        temp_data_y = data_y
    end

    train_x = temp_data_x[:, 1:split_ind]
    test_x = temp_data_x[:, split_ind+1:end]
    train_y = temp_data_y[1:split_ind]
    test_y = temp_data_y[split_ind+1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end

"""
Sequential loading and ratio split of the data.

# Arguments
$ARG_DATA_X
$ARG_DATA_Y
$ARG_RATIO
- `seq_ind::Array`:: the sequence IDs.
"""
function DataSplit(data_x::RealMatrix, data_y::RealVector, ratio::Real, seq_ind::Array)
    dim, n_data = size(data_x)
    n_splits = length(seq_ind)

    train_x = Array{Float64}(undef, dim, 0)
    train_y = Array{Float64}(undef, 0)
    test_x = Array{Float64}(undef, dim, 0)
    test_y = Array{Float64}(undef, 0)

    # Iterate over all splits
    for ind in seq_ind
        local_x = data_x[:, ind[1]:ind[2]]
        local_y = data_y[ind[1]:ind[2]]
        # n_data = ind[2] - ind[1] + 1
        n_data = size(local_x)[2]
        split_ind = Int(floor(n_data*ratio))

        train_x = [train_x local_x[:, 1:split_ind]]
        test_x = [test_x local_x[:, split_ind+1:end]]
        train_y = [train_y; local_y[1:split_ind]]
        test_y = [test_y; local_y[split_ind+1:end]]
    end
    return DataSplit(train_x, test_x, train_y, test_y)
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Loads the iris dataset for testing and examples.

# Arugments
- `data_path::AbstractString`: the path to the dataset.
$ARG_RATIO
"""
function load_iris(data_path::AbstractString ; ratio::Real = 0.8)
    raw_data = readdlm(data_path,',')
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    raw_x = convert(Array{Float, 2}, raw_data[2:end, 2:5])
    raw_y_labels = raw_data[2:end, 6]
    raw_y = Array{Int}(undef, 0)
    for ix in eachindex(raw_y_labels)
        for jx in eachindex(labels)
            if raw_y_labels[ix] == labels[jx]
                push!(raw_y, jx)
            end
        end
    end
    n_samples, n_features = size(raw_x)

    # Julia is column-major, so use columns for features
    raw_x = permutedims(raw_x)

    # Shuffle the data and targets
    ind_shuffle = Random.randperm(n_samples)
    x = raw_x[:, ind_shuffle]
    y = raw_y[ind_shuffle]

    data = DataSplit(x, y, ratio)

    return data
end

# """
#     get_cvi_data(data_file::String)

# Get the cvi data specified by the data_file path.
# """
# function get_cvi_data(data_file::AbstractString)
#     # Parse the data
#     data = readdlm(data_file, ',')
#     data = permutedims(data)
#     train_x = convert(Matrix{Float}, data[1:2, :])
#     train_y = convert(Vector{Int}, data[3, :])

#     return train_x, train_y
# end
