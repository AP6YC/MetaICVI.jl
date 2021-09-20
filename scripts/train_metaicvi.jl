using MetaICVI
# using ScikitLearn: fit!, score, @sk_import
# using ScikitLearn: fit!, score
using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn
# using ProgressMeter
using PyCallJLD
# @sk_import linear_model:RidgeClassifier
# learner = RidgeClassifier()

# Include local utils
# include("../test/test_utils.jl")

# Identify the rocket file
# rocket_file = "data/models/rocket.jld2"
# classifier_file = "data/models/classifier.jld"

# # Point to data
# data_dir(args...) = joinpath("data/training", args...)
data_dir = "data/training"

# Create the options
opts = MetaICVIOpts(
    # rocket_file = rocket_file,
    # classifier_file = classifier_file,
    n_rocket = 20
)

# Create the metaicvi object
metaicvi = MetaICVIModule(opts)

(features_data, features_targets) = get_training_features(metaicvi, data_dir)

# Create a split for training/testing the learner, correct for how scikitlearn expects features
x_train, x_test, y_train, y_test = train_test_split(transpose(features_data), features_targets, test_size=0.4)

# Train and test the learner
# fit!(metaicvi.classifier, x_train, y_train)
train_and_save(metaicvi, x_train, y_train)
performance = score(metaicvi.classifier, x_test, y_test)

@info "Performance is: $performance"
