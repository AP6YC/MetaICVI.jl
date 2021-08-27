module MetaICVI

# Common types and method
include("common.jl")

# Meta-icvi module definition
include("meta-icvi/meta-icvi.jl")

export

    MetaICVIModule,
    Rocket,
    RocketKernel,
    apply_kernel,
    apply_kernels

end
