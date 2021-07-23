using Oceananigans.Grids: Center, Face

linear(x) = 0.5x - 3
quadratic(x) = 0.5x^2 -3x + 5

@testset "Coarse Graining" begin
    @testset "Center" begin
        for func in [linear, quadratic]
            @testset "$func" begin
                x = range(0, 100, length=100)
                y = func.(x)

                N = 20
                y_coarse = coarse_grain(y, N, Center())

                @test length(y_coarse) == N
                @test mean(y_coarse) ≈ mean(y_coarse)

                if func == linear
                    @test all(diff(y_coarse) .≈ diff(y_coarse)[1])
                end
            end
        end
    end

    @testset "Face" begin
        for func in [linear, quadratic]
            @testset "$func" begin
                # 102 -> 22 tests the case where the interior can be easily coarse grained like `Center`.
                # 129 -> 33 tests the case where we need to interpolate between grid cells.
                lengths = [(102, 22), (129, 33)]

                for (n, N) in lengths
                    x = range(0, 100, length=n)
                    y = func.(x)

                    y_coarse = coarse_grain(y, N, Face())

                    y_interior = y[2:end-1]
                    y_coarse_interior = y_coarse[2:end-1]

                    @test length(y_coarse) == N
                    @test mean(y_interior) ≈ mean(y_coarse_interior)
                end
            end
        end
    end
end
