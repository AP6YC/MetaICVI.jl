using MetaICVI
using Test

include("test_utils.jl")

@testset "MetaICVI.jl" begin
    # Write your tests here.
end

@testset "1: Iris Training and Correlation" begin
    include("1_correleation_iris.jl")
end

@testset "2: ICVI-Spearman" begin
    include("2_icvi-spearman.jl")
end
