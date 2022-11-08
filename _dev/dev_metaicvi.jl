# using Revise
using NumericalTypeAliases
using MetaICVI
using Logging

# Include some test utilities (data loading, etc.)
include("../test/test_utils.jl")

# Point to the correct data directories
data_dir(args...) = joinpath("data", args...)
training_dir(args...) = data_dir("training", args...)
testing_dir(args...) = data_dir("testing", args...)
models_dir(args...) = data_dir("models", args...)
results_dir(args...) = joinpath("data/results", args...)

classifier_file = models_dir("classifier.jld")
rocket_file = models_dir("rocket.jld2")

# Cleanup
@info "--- REMOVING FILES ---"
isfile(classifier_file) && rm(classifier_file)
isfile(rocket_file) && rm(rocket_file)

@info "--- CREATING FIRST MODULE ---"
# Create the module
opts = MetaICVIOpts(
    # fail_on_missing = true
    fail_on_missing = false
)
metaicvi = MetaICVIModule(opts)

# Train and save
@info "--- TRAINING FIRST MODULE ---"
features_data, features_targets = get_training_features(metaicvi, training_dir())

data = DataSplit(features_data, features_targets, 0.8, shuffle=true)
train_and_save(metaicvi, data.train_x, data.train_y)

@info "--- CREATING SECOND MODULE ---"
# Create the module
opts = MetaICVIOpts(
    fail_on_missing = true
)
new_metaicvi = MetaICVIModule(opts)

# Load the data and test across all supervised modules
# data = load_iris(testing_dir("Iris.csv"))
# data.train_y = relabel_cvi_data(data.train_y)

# Iterate over the data
@info "--- TESTING ---"
n_data = length(data.test_y)
performances = zeros(n_data)
for i = 1:n_data
    sample = data.test_x[:, i]
    label = data.test_y[i]
    # performances[i] = get_metaicvi(new_metaicvi, sample, label)
    performances[i] = get_metaicvi(metaicvi, sample, label)
end

# Cleanup
@info "--- CLEAN UP ---"
rm(classifier_file)
rm(rocket_file)

@info "Max perf: $(maximum(performances))"
