using PyCall, JLD, PyCallJLD

using ScikitLearn
using ScikitLearn.Pipelines

@sk_import decomposition: PCA
@sk_import linear_model: LinearRegression

pip = JLD.load("pipeline.jld", "pip") # Load back the pipeline