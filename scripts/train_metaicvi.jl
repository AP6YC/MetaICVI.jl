using MetaICVI
using ScikitLearn: fit!, score, @sk_import
using ScikitLearn.CrossValidation: train_test_split

@sk_import linear_model:RidgeClassifier

learner = RidgeClassifier()

# Include local utils
include("../test/test_utils.jl")

# Identify the rocket file
rocket_file = "default_rocket.jld2"

# Point to data
data_dir(args...) = joinpath("data/training", args...)

# Create the options
opts = MetaICVIOpts(
    rocket_file = rocket_file
)

# Create the metaicvi object
metaicvi = MetaICVIModule(opts)

# Load the data
correct_x, correct_y = get_cvi_data(data_dir("correct_partition.csv"))
under_x, under_y = get_cvi_data(data_dir("under_partition.csv"))
over_x, over_y = get_cvi_data(data_dir("over_partition.csv"))

# Package the data conveniently
data = Dict(
    "correct" => Dict(
        "x" => correct_x,
        "y" => correct_y
    ),
    "under" => Dict(
        "x" => under_x,
        "y" => under_y
    ),
    "over" => Dict(
        "x" => over_x,
        "y" => over_y
    )
)

# Create the target containers
features_dim = metaicvi.opts.n_rocket
data_lengths = [length(correct_y), length(under_y), length(over_y)]
offset_lengths = [0, length(correct_y), length(under_y)]
data_length = sum(data_lengths)
features_data = zeros(features_dim, data_length)
features_targets = zeros(Int, data_length)

# Mapping of datatype to numeral target for classification
type_to_num = Dict(
    "correct" => 1,
    "under" => 2,
    "over" => 3
)

# Itereate over all data to get features
for (type, subdata) in data
    for i = 1:length(subdata["y"])
        # Extract the sample and label
        sample = subdata["x"][:, i]
        label = subdata["y"][i]

        # Compute and retrieve the features
        get_metaicvi(metaicvi, sample, label)
        features = metaicvi.features

        # Save the results
        # Get the offset directly from the type mapping and explicit definition
        data_offset = offset_lengths[type_to_num[type]]
        features_data[:, i + data_offset] = features
        features_targets[i + data_offset] = type_to_num[type]
    end
end

# Create a split for training/testing the learner, correct for how scikitlearn expects features
x_train, x_test, y_train, y_test = train_test_split(transpose(features_data), features_targets, test_size=0.4)

fit!(learner, x_train, y_train)
score(learner, x_test, y_test)
