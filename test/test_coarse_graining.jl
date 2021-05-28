using Oceananigans.Grids: Center, Face
using OceanParameterizations.DataWrangling: coarse_grain_linear_interpolation

@testset "Coarse Graining" begin
    @testset "Center" begin
        linear(x, m, c) = m * x + c

        y_linear = Array(linear.(1:100, 0.5, -3))
        N = 20
        
        y_linear_coarse = coarse_grain(y_linear, N, Center)

        @test length(y_linear_coarse) == N
        @test all(diff(y_linear_coarse) .≈ diff(y_linear_coarse)[1])
        @test mean(y_linear) ≈ mean(y_linear_coarse)
    end

    @testset "Face" begin
        linear(x, m, c) = m * x + c
        quadratic(x, a, b, c) = a * x ^ 2 + b * x + c
        
        
        y_linear = Array(linear.(1:100, 0.5, -3))
        y_quadratic = Array(quadratic.(1:100, 0.5, -3, 5))
        N = 20
        
        y_linear_coarse = coarse_grain_linear_interpolation(y_linear, N, Face)
        y_linear_quadratic = coarse_grain_linear_interpolation(y_quadratic, N, Face)

        @test length(y_linear_coarse) == N
        @test all(diff(y_linear_coarse) .≈ diff(y_linear_coarse)[1])
        @test mean(y_linear) ≈ mean(y_linear_coarse)

        @test length(y_linear_quadratic) == N
        @test_skip mean(y_quadratic[2:end-1]) ≈ mean(y_linear_quadratic[2:end-1])
    end
end