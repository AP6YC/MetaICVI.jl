# using PyCall
# using JLD
using PyCallJLD
using MetaICVI
using ClusterValidityIndices
using Test
using Logging

# Include some test utilities (data loading, etc.)
include("test_utils.jl")

# MetaICVI module testing
@testset "MetaICVI.jl" begin
    # Create the module
    opts = MetaICVIOpts(
        # fail_on_missing = true
        fail_on_missing = false
    )
    metaicvi = MetaICVIModule(opts)

    # Display some aspects of the module
    @info fieldnames(typeof(metaicvi))
    @info metaicvi
    @info metaicvi.classifier

    # Point to the correct data directories
    data_dir(args...) = joinpath("../data/testing", args...)
    results_dir(args...) = joinpath("../data/results", args...)

    # Load the data and test across all supervised modules
    data = load_iris(data_dir("Iris.csv"))
    # data.train_y = relabel_cvi_data(data.train_y)

    # Iterate over the data
    n_data = length(data.train_y)
    performances = zeros(n_data)
    for i = 1:n_data
        sample = data.train_x[:, i]
        label = data.train_y[i]
        performances[i] = get_metaicvi(metaicvi, sample, label)
    end

    # Perform some simple tests
    @test all(performances .>= 0)
    @test all(performances .<= 1)
end

# Rocket testing
@testset "Rocket.jl" begin
    # Test saving and loading
    filepath = "my_rocket"

    # Create the rocket module
    my_rocket = MetaICVI.Rocket.RocketModule()

    # Save the module to the filepath
    MetaICVI.Rocket.save_rocket(my_rocket, filepath)

    # Test that the file exists
    @test isfile(filepath)

    # Load the rocket module
    my_new_rocket = MetaICVI.Rocket.load_rocket(filepath)

    # Delete the saved file
    rm(filepath)
end
