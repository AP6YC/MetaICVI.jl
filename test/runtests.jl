"""
    runtests.jl

The entry point to unit tests for the MetaICVI.jl package.
"""

using SafeTestsets

@safetestset "All Test Sets" begin
    include("test_sets.jl")
end # @safetestset "All Test Sets"


# include("test_sets.jl")
