using StatsBase
using ClusterValidityIndices
using Logging
using Plots
using MetaICVI
using Random

# Plotting options
dpi = 300       # Plotting dots-per-inch
theme(:dark)    # Plotting style
# gr()            # GR backend (default for Plots.jl)
unicodeplots()

# Include the library definitions
# include(projectdir("julia/lib_sim.jl"))
# include("rocket.jl")

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)
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
    cSIL(),
    PS()
    # PS(),
    # GD43()
]
n_cvis = length(cvis)

# Setup the correlation
n = 100                         # Window size
n_corr = n_samples - n + 1      # Number of correlations
corrs = zeros(n_corr, n_data)   # Preallocation of correlations
iter_corr = n:n_samples         # Iterator for the correlation generation

plot_labels = zeros(n_samples, n_data)

# Iterate over all partition types
for dx = 1:n_data
    data_path = data_paths[dx]

    # Load the training data
    data, labels = get_cvi_data(data_path)

    plot_labels[:, dx] = labels

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

n_window = 100
n_kernels = 2
n_rocket = n_corr - n_window + 1
# corrs = zeros(n_corr, n_data)   # Preallocation of correlations
rocket_features = zeros(n_rocket, n_kernels, 2, n_data)
# rocket_features = []
# rocket_maxes = zeros(n_rocket, n_data)
iter_rocket = n + n_window:n_samples
@info n_rocket
@info iter_rocket

rocket = Rocket(n_window, n_kernels)
for dx = 1:n_data
    # for i = iter_rocket
    for i = 1:size(corrs)[1]
        if i >= n_window
            # sample_window = corrs[i - n_window + 1: i, dx]
            sample_window = corrs[i - n_window + 1: i, dx]
            # push!(rocket_features, apply_kernels(rocket, sample_window))
            # rocket_features[i-n_window-n+1, :, :, dx] = apply_kernels(rocket, sample_window)
            # index, ppv, max, data
            rocket_features[i - n_window + 1, :, :, dx] = apply_kernels(rocket, sample_window)
        end
    end
end

# Plot
p = plot(dpi=dpi, legend=:bottomleft)
for i = 1:n_data
    plot!(p, iter_corr, corrs[:, i], label=basename(data_paths[i]))
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

g = plot(dpi=dpi, legend=:bottomleft)
for i = 1:n_data
    plot!(g, plot_labels[:, i], label=basename(data_paths[i]))
end

h = plot(dpi=dpi, legend=:bottomleft)
for i = 1:n_data
    # for j = 1:size(rocket_features[i])[1]
    for j = 1:n_kernels
        # plot!(h, rocket_features[i][j, 1], label=basename(data_paths[i]))
        plot!(h, 199:2000, rocket_features[:, j, 2, i], label=basename(data_paths[i]))
    end
end

l = @layout [a; b; c]
# pt = plot(p, g, layout = l, size=(500,500))
pt = plot(p, g, h, layout = l, size=(800,500))

# Save and show the plot
# png(pt, results_dir("4_corr_rocket"))
display(pt)
