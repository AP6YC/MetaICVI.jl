using StatsBase
using AdaptiveResonance
using ClusterValidityIndices
using Logging
using Plots
using Random

# Plotting options
dpi = 300       # Plotting dots-per-inch
theme(:dark)    # Plotting style
gr()            # GR backend (default for Plots.jl)

# Parameters
n = 5                           # Window size

# Include the library definitions
# include(projectdir("julia/lib_sim.jl"))
include("test_utils.jl")

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)
# data_dir(args...) = projectdir("work/data/meta_icvi", args...)
# results_dir(args...) = projectdir("work/results/meta_icvi", args...)
data_dir(args...) = joinpath("../data/testing", args...)
results_dir(args...) = joinpath("../data/results", args...)

# Load the data and test across all supervised modules
data = load_iris(data_dir("Iris.csv"))
data.train_x = convert(Array{Float64, 2}, data.train_x)
data.train_y = convert(Array{Int64, 1}, data.train_y)

# data.train_x, data.train_y = sort_cvi_data(convert(Matrix{Real}, data.train_x), data.train_y)
data.test_x, data.test_y = sort_cvi_data(convert(Matrix{Real},data.test_x), data.test_y)
# data.test_x, data.test_y = sort_cvi_data(data.test_x, data.test_y)

# Initialize the ART module
art = DDVFA()
# Set up the data manually because the module can't infer from single samples
# data_setup!(art.config, data.train_x)

# Get the dimension and size of the data
dim, n_samples = AdaptiveResonance.get_data_shape(data.train_x)
y_hat_train = zeros(Int64, n_samples)
_, n_samples_test = AdaptiveResonance.get_data_shape(data.test_x)
y_hat = zeros(Int64, n_samples_test)

# Create the CVIs
cvis = [
    PS(),
    GD43()
]
cvis_test = deepcopy(cvis)
n_cvis = length(cvis)

# Train the DDVFA model
y_hat_train = train!(art, data.train_x, y=data.train_y)
# y_hat_train = train!(art, train_x)
y_hat = classify(art, data.test_x)

# Preallocate the criterion_values
criterion_values = zeros(n_samples, n_cvis)
criterion_values_test = zeros(n_samples_test, n_cvis)
labels_ordered = relabel_cvi_data(y_hat_train)
labels_ordered_test = relabel_cvi_data(y_hat)
for i = 1:n_samples
    sample = convert(Array{Float64, 1}, data.train_x[:, i])
    label = labels_ordered[i]
    for j = 1:n_cvis
        criterion_values[i, j] = get_icvi!(cvis[j], sample, label)
    end
end
for i = 1:n_samples_test
    sample = convert(Array{Float64, 1}, data.test_x[:, i])
    label = labels_ordered_test[i]
    for j = 1:n_cvis
        criterion_values_test[i, j] = get_icvi!(cvis_test[j], sample, label)
    end
end

# Get the train correlations
n_corr = n_samples - n + 1      # Number of correlations
corrs = zeros(n_corr)           # Preallocation for correlations
iter_corr = n:n_samples         # Iterator for the correlation generation
for i = iter_corr
    corrs[i-n+1] = corspearman(criterion_values[i-n+1:i, 1], criterion_values[i-n+1:i, 2])
end

# Get the test correlations
n_corr_test = n_samples_test - n + 1      # Number of correlations
corrs_test = zeros(n_corr_test)           # Preallocation for correlations
iter_corr_test = n:n_samples_test         # Iterator for the correlation generation
for i = iter_corr_test
    corrs_test[i-n+1] = corspearman(criterion_values_test[i-n+1:i, 1], criterion_values_test[i-n+1:i, 2])
end

# Plot the two incremental trends ("manual" and porcelain) atop one another
p = plot(dpi=dpi, legend=:bottomleft)
plot!(p, iter_corr, corrs, label="Spearman")
plot!(p, 1:n_samples, y_hat_train, label="Labels")
plot!(p, 1:n_samples, data.train_y, label="True Labels")
# plot!(p, 1:n_corr, criterion_values_p, label="Porcelain")
title!("DDVFA + Iris Train")
xlabel!("Sample Index")
ylabel!("Spearman")

# Plot the two incremental trends ("manual" and porcelain) atop one another
g = plot(dpi=dpi, legend=:bottomleft)
plot!(g, iter_corr_test, corrs_test, label="Spearman")
plot!(g, 1:n_samples_test, y_hat, label="Labels")
plot!(g, 1:n_samples_test, data.test_y, label="True Labels")
# plot!(p, 1:n_corr, criterion_values_p, label="Porcelain")
title!("DDVFA + Iris Test")
xlabel!("Sample Index")
ylabel!("Spearman")

# Show the plots together
l = @layout [a; b]
# pt = plot(p, g, layout = l, size=(500,500))
pt = plot(p, g, layout = l, size=(800,500))

# Save and show the plot
png(pt, results_dir("1_correlation_iris"))
# display(pt)

# xlims!(1, n_corr)
# ylims!(0, Inf)

# Calculate performance
perf_train = performance(y_hat_train, data.train_y)
perf_test = performance(y_hat, data.test_y)

@info "DDVFA Training Perf: $perf_train"
@info "DDVFA Testing Perf: $perf_test"
