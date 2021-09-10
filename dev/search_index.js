var documenterSearchIndex = {"docs":
[{"location":"man/contributing/#Contributing","page":"Contributing","title":"Contributing","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"To formally contribute to the package, please follow the usual branch pull request procedure:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"Fork the project.\nCreate your feature branch (git checkout -b my-new-feature).\nCommit your changes (git commit -am 'Added some feature').\nPush to the branch (git push origin my-new-feature).\nCreate a new GitHub pull request.","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"If you simply have suggestions for improvement, Sasha Petrenko (<sap625@mst.edu>) is the current developer and maintainer of the MetaICVI.jl package, so please feel free to reach out with thoughts and questions.","category":"page"},{"location":"man/full-index/#main-index","page":"Index","title":"Index","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"All structures and methods can be found in the Full Index with their accompanying docstrings in Documentation","category":"page"},{"location":"man/full-index/#Full-Index","page":"Index","title":"Full Index","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"","category":"page"},{"location":"man/full-index/#Documentation","page":"Index","title":"Documentation","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [\n    MetaICVI,\n    MetaICVI.Rocket\n]","category":"page"},{"location":"man/full-index/#MetaICVI.MetaICVIModule","page":"Index","title":"MetaICVI.MetaICVIModule","text":"MetaICVIModule\n\nStateful information for a single MetaICVI module.\n\nFields\n\nopts::MetaICVIOpts: options for construction.\ncvis::Vector{AbstractCVI}: list of cvis used for computing the CVIs.\ncriterion_values::RealVector: list of outputs of the cvis used for computing correlations.\ncorrelations::RealVector: list of outputs of the rank correlations.\nfeatures::RealVector: list of outputs of the rocket feature kernels.\nrocket::RocketModule: time-series random feature kernels module.\nclassifier::MetaICVIClassifier: ScikitLearn classifier.\nperformance::RealFP: final output of the most recent the Meta-ICVI step.\nis_pretrained::Bool: internal flag for if the classifier is trained and ready for inference.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#MetaICVI.MetaICVIModule-Tuple{MetaICVIOpts}","page":"Index","title":"MetaICVI.MetaICVIModule","text":"MetaICVIModule(opts::MetaICVIOpts)\n\nInstantiate a MetaICVIModule with given options.\n\nArguments\n\nopts::MetaICVIOpts: options struct for the MetaICVI object.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.MetaICVIModule-Tuple{}","page":"Index","title":"MetaICVI.MetaICVIModule","text":"MetaICVIModule()\n\nDefault constructor for the MetaICVIModule.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.MetaICVIOpts","page":"Index","title":"MetaICVI.MetaICVIOpts","text":"MetaICVIOpts()\n\nMeta-ICVI module options.\n\nExamples\n\njulia> MetaICVIOpts()\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#Base.show-Tuple{IO, MetaICVIModule}","page":"Index","title":"Base.show","text":"Base.show(io::IO, metaicvi::MetaICVIModule)\n\nDisplay a metaicvi module to the command line.\n\nArguments\n\nio::IO: default io stream.\nmetaicvi::MetaICVIModule: metaicvi object about which to display info.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.construct_classifier-Tuple{MetaICVIOpts}","page":"Index","title":"MetaICVI.construct_classifier","text":"construct_classifier(opts::MetaICVIOpts)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.get_correlations-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.get_correlations","text":"get_correlations(metaicvi::MetaICVIModule)\n\nCompute and store the rank correlations from the cvi values.\n\nArguments\n\nmetaicvi::MetaICVIModule: the Meta-ICVI module.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.get_icvis-Tuple{MetaICVIModule, AbstractVector{T} where T<:Real, Integer}","page":"Index","title":"MetaICVI.get_icvis","text":"get_icvis(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)\n\nCompute and store the icvi criterion values.\n\nArguments\n\nmetaicvi::MetaICVIModule: the Meta-ICVI module.\nsample::RealVector: the sample used for clustering.\nlabel::Integer: the label prescribed to the sample by the clustering algorithm.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.get_metaicvi-Tuple{MetaICVIModule, AbstractVector{T} where T<:Real, Integer}","page":"Index","title":"MetaICVI.get_metaicvi","text":"get_metaicvi(metaicvi::MetaICVIModule, sample::RealVector, label::Integer)\n\nCompute and return the meta-icvi value.\n\nArguments\n\nmetaicvi::MetaICVIModule: the Meta-ICVI module.\nsample::RealVector: the sample used for clustering.\nlabel::Integer: the label prescribed to the sample by the clustering algorithm.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.get_probability-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.get_probability","text":"get_probability(metaicvi::MetaICVIModule)\n\nCompute and store the metaicvi value from the classifier.\n\nArguments\n\nmetaicvi::MetaICVIModule: the Meta-ICVI module.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.get_rocket_features-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.get_rocket_features","text":"get_rocket_features(metaicvi::MetaICVIModule)\n\nCompute and store the rocket features.\n\nArguments\n\nmetaicvi::MetaICVIModule: the Meta-ICVI module.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.is_pretrained-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.is_pretrained","text":"is_pretrained(metaicvi::MetaICVIModule)\n\nChecks if the classifier is pretrained to permit inference.\n\nArguments\n\nmetaicvi::MetaICVIModule: metaicvi module containing the classifier to check.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.load_classifier-Tuple{String}","page":"Index","title":"MetaICVI.load_classifier","text":"load_classifier(filepath::String)\n\nLoad the classifier at the filepath.\n\nArguments\n\nfilepath::String: location of the classifier .jld file.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.safe_save_classifier-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.safe_save_classifier","text":"safe_save_classifier(metaicvi::MetaICVIModule)\n\nError handle saving of the metaicvi classifier.\n\nArguments\n\nmetaicvi::MetaICVIModule: metaicvi module containing the classifier and path for saving.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.safe_save_rocket-Tuple{MetaICVIModule}","page":"Index","title":"MetaICVI.safe_save_rocket","text":"safe_save_rocket(metaicvi::MetaICVIModule)\n\nError handle saving of the metaicvi rocket kernels.\n\nArguments\n\nmetaicvi::MetaICVIModule: metaicvi module containing the classifier and path for saving.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.save_classifier-Tuple{PyCall.PyObject, String}","page":"Index","title":"MetaICVI.save_classifier","text":"save_classifier(classifier::MetaICVIClassifier, filepath::String)\n\nSave the classifier at the filepath.\n\nArguments\n\nclassifier::MetaICVIClassifier: classifier object to save.\nfilepath::String: name/path to save the classifier .jld file.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.train_and_save-Tuple{MetaICVIModule, AbstractMatrix{T} where T<:Real, AbstractVector{T} where T<:Integer}","page":"Index","title":"MetaICVI.train_and_save","text":"train_and_save(metaicvi::MetaICVIModule, x::RealMatrix, y::IntegerVector)\n\nTrain the classifier on x/y and save the kernels and classifier.\n\nArguments\n\nmetaicvi::MetaICVIModule: metaicvi module to save with.\nx::RealMatrix: features to train on.\ny::IntegerVector: correct/over/under partition targets.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.Rocket.RocketKernel","page":"Index","title":"MetaICVI.Rocket.RocketKernel","text":"RocketKernel\n\nStructure containing information about one rocket kernel.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#MetaICVI.Rocket.RocketModule","page":"Index","title":"MetaICVI.Rocket.RocketModule","text":"RocketModule\n\nStructure containing a vector of rocket kernels.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#MetaICVI.Rocket.RocketModule-Tuple{Integer, Integer}","page":"Index","title":"MetaICVI.Rocket.RocketModule","text":"RocketModule(input_length::Integer, n_kernels::Integer)\n\nCreate a new RocketModule structure, requiring feature length and the number of kernels.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.Rocket.RocketModule-Tuple{}","page":"Index","title":"MetaICVI.Rocket.RocketModule","text":"RocketModule()\n\nDefault constructor for the RocketModule object.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.Rocket.apply_kernel-Tuple{MetaICVI.Rocket.RocketKernel, AbstractVector{T} where T<:Real}","page":"Index","title":"MetaICVI.Rocket.apply_kernel","text":"apply_kernel(kernel::RocketKernel, x::RealVector)\n\nApply a single RocketModule kernel to the sequence x.\n\nArguments\n\nkernel::RocketKernel: rocket kernel used for computing features.\nx::RealVector: data sequence for computing rocket features.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.Rocket.apply_kernels-Tuple{MetaICVI.Rocket.RocketModule, AbstractVector{T} where T<:Real}","page":"Index","title":"MetaICVI.Rocket.apply_kernels","text":"apply_kernels(rocket::RocketModule, x::RealVector)\n\nRun a vector of rocket kernels along a sequence x.\n\nArguments\n\nrocket::RocketModule: rocket module containing many kernels for processing.\nx::RealVector: data sequence for computing rocket features.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#MetaICVI.Rocket.load_rocket","page":"Index","title":"MetaICVI.Rocket.load_rocket","text":"load_rocket(filepath::String=\"rocket.jld2\")\n\nLoad and return a rocket module with existing parameters from a .jld2 file.\n\nArguments\n\nfilepath::String: path to .jld2 containing rocket parameters. Defaults to rocket.jld2.\n\n\n\n\n\n","category":"function"},{"location":"man/full-index/#MetaICVI.Rocket.save_rocket","page":"Index","title":"MetaICVI.Rocket.save_rocket","text":"save_rocket(rocket::RocketModule, filepath::String=\"rocket.jld2\")\n\nSave the rocket parameters to a .jld2 file.\n\nArguments\n\nrocket::RocketModule: rocket module to save. filepath::String: path to .jld2 for saving rocket parameters. Defaults to rocket.jld2.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#Package-Guide","page":"Guide","title":"Package Guide","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The MetaICVI.jl package is built upon modules that contain all of the state information during training and inference. The MetaICVI modules are driven by options, which are themselves mutable keyword argument structs from the Parameters.jl package.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"To work with MetaICVI.jl, you should know:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"How to install the package\nMetaICVI module basics\nHow to use MetaICVI module options","category":"page"},{"location":"man/guide/#installation","page":"Guide","title":"Installation","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The MetaICVI.jl package can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"pkg> add MetaICVI","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Alternatively, it can be added to your environment in a script with","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"using Pkg\nPkg.add(\"MetaICVI\")","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"If you wish to have the latest changes between releases, you can directly add the GitHub repo as a dependency with","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"pkg> add https://github.com/AP6YC/MetaICVI.jl","category":"page"},{"location":"man/guide/#metaicvi_modules","page":"Guide","title":"MetaICVI Modules","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"TODO","category":"page"},{"location":"man/guide/#metaicvi_options","page":"Guide","title":"MetaICVI Options","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"TODO","category":"page"},{"location":"man/examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"man/examples/","page":"Examples","title":"Examples","text":"There are examples for every structure in the package within the package's examples/ folder. The code for several of these examples is provided here.","category":"page"},{"location":"man/examples/","page":"Examples","title":"Examples","text":"TODO","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MetaICVI","category":"page"},{"location":"#MetaICVI","page":"Home","title":"MetaICVI","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MetaICVI.","category":"page"},{"location":"","page":"Home","title":"Home","text":"These pages serve to summarize the internals and functionality of the Meta-ICVI approach. This project was developed by the Applied Computational Intelligence Laboratory (ACIL) of the Missouri University of Science and Technology (S&T) under the supervision of Teledyne Technologies for the DARPA L2M program.","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the Index for the complete list of documented functions and types.","category":"page"},{"location":"#Manual-Outline","page":"Home","title":"Manual Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This documentation is split into the following sections:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"man/guide.md\",\n    \"man/examples.md\",\n    \"man/contributing.md\",\n    \"man/full-index.md\",\n]\nDepth = 1","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Package Guide provides a tutorial to the full usage of the package, while Examples gives sample workflows using a variety of ART modules.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Instructions on how to contribute to the package are found in Contributing, and docstrings for every element of the package is listed in the Index.","category":"page"}]
}
