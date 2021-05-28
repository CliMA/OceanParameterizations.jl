@testset "Zero Mean Unit Variance Scaling" begin
    random_datas = [rand(10), rand(5,5), rand(3,3,3)]
    
    for random_data in random_datas
        scaling = ZeroMeanUnitVarianceScaling(random_data)
        @test scaling.μ == mean(random_data)
        @test scaling.σ == std(random_data)
        @test mean(scaling.(random_data)) ≈ 0 atol=1e-10
        @test std(scaling.(random_data)) ≈ 1
        @test all(inv(scaling).(scaling.(random_data)) .≈ random_data)
    end
end

@testset "Min-Max Scaling" begin
    random_datas = [rand(10), rand(5,5), rand(3,3,3)]
    
    ranges = ((a=0, b=10), (a=-5, b=-1), (a=1, b=2), (a=-1, b=20))

    for random_data in random_datas, range in ranges
        a = range.a
        b = range.b
        scaling = MinMaxScaling(random_data, a=a, b=b)
        @test scaling.data_min == minimum(random_data)
        @test scaling.data_max == maximum(random_data)
        @test scaling.a == a
        @test scaling.b == b
        @test minimum(scaling.(random_data)) ≈ a atol=1e-10
        @test maximum(scaling.(random_data)) ≈ b atol=1e-10
        @test all(inv(scaling).(scaling.(random_data)) .≈ random_data)
    end
end