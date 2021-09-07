using MetaICVI
# using MLJ

# Include local utils
include("../test/test_utils.jl")

# Identify the rocket file
rocket_file = "default_rocket.jld2"

# Point to data
data_dir(args...) = joinpath("data/training", args...)

# Create the learner
# classifier = (@load RidgeClassifier)()
# my_machine = machine(classifier, )
# Create the options
opts = MetaICVIOpts(
    rocket_file = rocket_file
)

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

# Create the metaicvi object
metaicvi = MetaICVIModule(opts)
for (type, subdata) in data
    for i = 1:length(correct_y)
        get_metaicvi(metaicvi, subdata["x"][:, i], subdata["y"][i])
        features = metaicvi.features
        @info features
    end
end
