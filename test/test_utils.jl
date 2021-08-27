using DelimitedFiles
using Logging
using Random
using StatsBase
using Statistics

"""
    DataSplit

A basic struct for encapsulating the four components of supervised training.
"""
mutable struct DataSplit
    train_x::Array
    test_x::Array
    train_y::Array
    test_y::Array
    DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Float)

Return a DataSplit struct that is split by the ratio (e.g. 0.8).
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real)
    dim, n_data = size(data_x)
    split_ind = Int(floor(n_data*ratio))

    train_x = data_x[:, 1:split_ind]
    test_x = data_x[:, split_ind+1:end]
    train_y = data_y[1:split_ind]
    test_y = data_y[split_ind+1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

Sequential loading and ratio split of the data.
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)
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
end # DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

"""
    load_iris(data_path::String ; split_ratio::Real = 0.8)

Loads the iris dataset for testing and examples.
"""
function load_iris(data_path::String ; split_ratio::Real = 0.8)
    raw_data = readdlm(data_path,',')
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    raw_x = raw_data[2:end, 2:5]
    raw_y_labels = raw_data[2:end, 6]
    raw_y = Array{Int64}(undef, 0)
    for ix = 1:length(raw_y_labels)
        for jx = 1:length(labels)
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

    data = DataSplit(x, y, split_ratio)

    return data
end # load_iris(data_path::String ; split_ratio::Real = 0.8)

"""
    get_cvi_data(data_file::String)

Get the cvi data specified by the data_file path.
"""
function get_cvi_data(data_file::String)
    # Parse the data
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = convert(Matrix{Float64}, data[1:2, :])
    train_y = convert(Vector{Int64}, data[3, :])

    return train_x, train_y
end # get_cvi_data(data_file::String)
