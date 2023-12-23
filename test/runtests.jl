using NeuralCollaborativeFiltering
using Test

@testset "NeuralCollaborativeFiltering.jl" begin

    ############## Recommendation quality metrics ##############

    @testset "reciprocal_rank" begin
        @test reciprocal_rank([5, 2, 1], [3, 5, 7]) ≈ 1/2
        @test reciprocal_rank([2, 6, 4], [1, 3, 5]) == 0.0
        @test reciprocal_rank([10], [10, 20, 30]) ≈ 1/1
        @test reciprocal_rank([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) == 0.25
        @test_throws MethodError reciprocal_rank([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])  # y <: Vector{Float64}
    end

    @testset "accuracy" begin
        @test accuracy([1, 2, 3], [1, 2, 3]) == 1.0
        @test accuracy([1, 2, 3], [1, 4, 3]) ≈ 2/3
        @test accuracy([4, 5, 6], [7, 8, 9]) == 0.0
        @test accuracy([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.2857142857142857
        @test_throws MethodError accuracy([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

    @testset "average_precision" begin
        @test average_precision([1, 2, 3], [1, 2, 3]) == 1.0
        @test average_precision([2, 6, 4], [1, 3, 5]) == 0.0
        @test average_precision([4, 2, 1], [1, 2, 3]) ≈ 2/3
        @test average_precision([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.7486394557823128
        @test_throws MethodError average_precision([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

    @testset "extended_reciprocal_rank" begin
        @test extended_reciprocal_rank([1, 2, 4, 7], [7, 8, 1, 5]) ≈ 1/3
        @test extended_reciprocal_rank([1, 2, 4], [4, 2, 3]) ≈ 2/3
        @test extended_reciprocal_rank([10, 20, 4], [1, 2, 3]) == 0.0
        @test extended_reciprocal_rank([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.6547619047619049
        @test_throws MethodError extended_reciprocal_rank([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

end
