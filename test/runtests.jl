using NeuralCollaborativeFiltering
using Test

@testset "NeuralCollaborativeFiltering.jl" begin
    # Write your tests here.


    # Recommendation quality metrics
    y = [8, 0, -1, 5, 4, 6, 7]
    ŷ = [0, -1, 3, 8, 4, 5, 7]
    @test accuracy(y, ŷ) ≈ 0.2857142857142857
    @test reciprocal_rank(y, ŷ) == 0.25
    @test average_precision(y, ŷ) ≈ 0.7486394557823128
    @test extended_reciprocal_rank(y, ŷ) ≈ 0.763888888888889

    y = [8, 0, -1, 5, 4.0, 6, 7] # ::Vector{Float64}
    @test_throws MethodError accuracy(y, ŷ)
    @test_throws MethodError reciprocal_rank(y, ŷ)
    @test_throws MethodError average_precision(y, ŷ)
    @test_throws MethodError extended_reciprocal_rank(y, ŷ)

    

end
