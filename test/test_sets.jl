# using PyCall
# using JLD
# using PyCallJLD
using MetaICVI
using ClusterValidityIndices
using Test

using Logging

@info @__MODULE__
# @info varinfo()

include("test_utils.jl")

@testset "MetaICVI.jl" begin
    # Write your tests here.
end

@testset "0: Module" begin
    # using PyCallJLD
    # using MetaICVI
    # Create the module
    opts = MetaICVIOpts(
        # rocket_file = "../data/models/rocket.jld2",
        # classifier_file = "../data/models/classifier.jld",
        fail_on_missing = true
    )
    metaicvi = MetaICVIModule(opts)
    @info fieldnames(typeof(metaicvi))
    @info metaicvi
    @info metaicvi.classifier

    # Point to the correct data directories
    data_dir(args...) = joinpath("../data/testing", args...)
    results_dir(args...) = joinpath("../data/results", args...)

    # Load the data and test across all supervised modules
    data = load_iris(data_dir("Iris.csv"))
    data.train_y = relabel_cvi_data(data.train_y)

    # Iterate over the data
    for i = 1:length(data.train_y)
        sample = data.train_x[:, i]
        label = data.train_y[i]
        # performance = get_metaicvi(metaicvi, sample, label)
    end
end

# Rocket testing
@testset "Rocket.jl" begin
    # Test saving and loading
    filepath = "my_rocket"
    my_rocket = MetaICVI.Rocket.RocketModule()
    MetaICVI.Rocket.save_rocket(my_rocket, filepath)
    my_new_rocket = MetaICVI.Rocket.load_rocket(filepath)
    rm(filepath)
end
