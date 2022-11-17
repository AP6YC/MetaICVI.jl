# using PyCall
# using JLD
# using ClusterValidityIndices
using
    MetaICVI,
    PyCallJLD,
    Test,
    Logging,
    NumericalTypeAliases

# Include some test utilities (data loading, etc.)
include("test_utils.jl")

# MetaICVI module testing
@testset "MetaICVI.jl" begin
    # Point to the correct data directories
    data_dir(args...) = joinpath("../data", args...)
    training_dir(args...) = data_dir("training", args...)
    testing_dir(args...) = data_dir("testing", args...)
    models_dir(args...) = data_dir("models", args...)
    results_dir(args...) = joinpath("../data/results", args...)

    # Create the module
    opts = MetaICVIOpts(
        # fail_on_missing = true
        fail_on_missing = false
    )
    metaicvi = MetaICVIModule(opts)

    # Train and save
    local_data = MetaICVI.load_training_data(training_dir())
    train_data, test_data = MetaICVI.split_training_data(local_data)
    test_x, test_y = MetaICVI.serialize_data(test_data)
    features_data, features_targets = get_training_features(metaicvi, train_data)
    train_and_save(metaicvi, features_data, features_targets)

    # Display some aspects of the module
    @info fieldnames(typeof(metaicvi))
    @info metaicvi
    @info metaicvi.classifier

    # Create the module
    opts = MetaICVIOpts(
        fail_on_missing = true
    )
    new_metaicvi = MetaICVIModule(opts)

    # Load the data and test across all supervised modules
    data = load_iris(testing_dir("Iris.csv"))
    # data.train_y = relabel_cvi_data(data.train_y)

    # Iterate over the data
    n_data = length(data.train_y)
    performances = zeros(n_data)
    performances_orig = zeros(n_data)
    for i = 1:n_data
        # sample = data.train_x[:, i]
        # label = data.train_y[i]
        # performances[i] = get_metaicvi(metaicvi, sample, label)
        sample = test_x[:, i]
        label = test_y[i]
        # performances[i] = get_metaicvi(new_metaicvi, sample, label)
        performances[i] = get_metaicvi(new_metaicvi, sample, label)
        performances_orig[i] = get_metaicvi(metaicvi, sample, label)
    end

    # Perform some simple tests
    @test all(performances .>= 0)
    @test all(performances .<= 1)
    @test all(performances_orig .>= 0)
    @test all(performances_orig .<= 1)

    # Cleanup
    rm(models_dir("classifier.jld"))
    rm(models_dir("rocket.jld2"))
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
