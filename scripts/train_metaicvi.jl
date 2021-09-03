using MetaICVI
using MLJ

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

# Load the data
correct_x, correct_y = get_cvi_data(data_dir("correct_partition.csv"))
under_x, under_y = get_cvi_data(data_dir("under_partition.csv"))
over_x, over_y = get_cvi_data(data_dir("over_partition.csv"))

# Create the metaicvi object
metaicvi = MetaICVIModule(opts)
for i = 1:length(correct_y)
    get_metaicvi(metaicvi, correct_x[:, i], correct_y[i])
    @info metaicvi.features
end
