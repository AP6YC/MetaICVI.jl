using StatsBase
using ClusterValidityIndices
using Logging
using Plots

# Plotting options
dpi = 300       # Plotting dots-per-inch
theme(:dark)    # Plotting style
# gr()            # GR backend (default for Plots.jl)
unicodeplots()

# Include the library definitions
# include(projectdir("julia/lib_sim.jl"))

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
# Random.seed!(0)
# data_dir(args...) = projectdir("work/data/meta_icvi", args...)
# results_dir(args...) = projectdir("work/results/meta_icvi", args...)
data_dir(args...) = joinpath("../data/training", args...)
results_dir(args...) = joinpath("../data/results", args...)

# Setup the data
data_paths = [
    data_dir("correct_partition.csv"),
    data_dir("over_partition.csv"),
    data_dir("under_partition.csv")
]
n_samples = 2000
n_data = 3

# Setup ICVIs
cvis = [
    # cSIL(),
    # PS()
    PS(),
    GD43()
]
n_cvis = length(cvis)

# Setup the correlation
n = 100                         # Window size
n_corr = n_samples - n + 1      # Number of correlations
corrs = zeros(n_corr, n_data)   # Preallocation of correlations
iter_corr = n:n_samples         # Iterator for the correlation generation

# Iterate over all partition types
for dx = 1:n_data
    data_path = data_paths[dx]

    # Load the training data
    data, labels = get_cvi_data(data_path)
    labels = relabel_cvi_data(labels)
    data_name = splitext(basename(data_path))[1]    # Get the data_name

    # Instantiate the icvis
    local_cvis = deepcopy(cvis)
    criterion_values = zeros(n_samples, n_cvis)

    # Run the ICVIs iteratively
    for i = 1:n_samples
        sample = data[:, i]
        label = labels[i]
        for j = 1:n_cvis
            criterion_values[i, j] = get_icvi!(local_cvis[j], sample, label)
        end
    end

    # Calculate the windowed correlations
    for i = iter_corr
        corrs[i-n+1, dx] = corspearman(criterion_values[i-n+1:i, 1], criterion_values[i-n+1:i, 2])
    end
end

p = plot(dpi=dpi, legend=:bottomleft)
for i = 1:n_data
    plot!(p, 1:n_corr, corrs[:, i], label=basename(data_paths[i]), size=(800,500))
end
# plot!(p, 1:n_samples, labels, label="Labels")
# plot!(p, 1:n_samples, data.train_y, label="True Labels")
# plot!(p, 1:n_corr, criterion_values_p, label="Porcelain")
# title!("Spearman, Data: " * basename(data_path))
title!("Partition ICVI Correlations")
xlabel!("Sample Index")
ylabel!("Spearman")
# xlims!(1, n_corr)
# ylims!(0, Inf)

# Save and show the plot
# png(p, results_dir("2_correlation_partition"))
display(p)
dest_file = results_dir("2_correlation_partition.txt")
savefig(p, dest_file)
