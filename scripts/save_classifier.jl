using JLD2
# using PyCallJLD
# using PyCall
using ScikitLearn: fit!, score, predict, @sk_import
@sk_import linear_model:RidgeClassifier

classifier = RidgeClassifier()

save_object("asdf.jld2", classifier)

my_new = load_object("asdf.jld2")
