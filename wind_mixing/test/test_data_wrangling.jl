using WindMixing: TrainingDatasets

using Oceananigans
using Oceananigans.OutputReaders
using JLD2

using Statistics: mean, std
using Oceananigans.Fields: Field, location
using Oceananigans.OutputReaders
using OceanParameterizations

@testset "Data Wrangling" begin
    dirnames = [
        "three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC",
        "three_layer_constant_fluxes_cubic_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_cubic"
    ]

    Nzs_coarse = [32, 64]
    multiple_datasets = [TrainingDatasets(dirnames, Nz_coarse=Nz) for Nz in Nzs_coarse]

    @testset "Coarse-Grained Grid Size" begin
        for i in 1:length(Nzs_coarse)
            Nz_coarse = Nzs_coarse[i]
            datasets = multiple_datasets[i]
            for data in datasets.data
                @test length(interior(data["u*"])[1,1,:,1]) == Nz_coarse
                @test length(interior(data["v*"])[1,1,:,1]) == Nz_coarse
                @test length(interior(data["T*"])[1,1,:,1]) == Nz_coarse

                @test length(interior(data["wu*"])[1,1,:,1]) == Nz_coarse + 1
                @test length(interior(data["wv*"])[1,1,:,1]) == Nz_coarse + 1
                @test length(interior(data["wT*"])[1,1,:,1]) == Nz_coarse + 1

                @test length(interior(data["∂u∂z*"])[1,1,:,1]) == Nz_coarse + 1
                @test length(interior(data["∂v∂z*"])[1,1,:,1]) == Nz_coarse + 1
                @test length(interior(data["∂T∂z*"])[1,1,:,1]) == Nz_coarse + 1

                @test data["u*"].grid.Nz == Nz_coarse
                @test data["v*"].grid.Nz == Nz_coarse
                @test data["T*"].grid.Nz == Nz_coarse

                @test data["wu*"].grid.Nz == Nz_coarse
                @test data["wv*"].grid.Nz == Nz_coarse
                @test data["wT*"].grid.Nz == Nz_coarse
            end
        end
    end

    @testset "Constant Surface Flux" begin
        for i in 1:length(Nzs_coarse)
            Nz_coarse = Nzs_coarse[i]
            datasets = multiple_datasets[i]
            for data in datasets.data
                @test all(interior(data["wu"])[1,1,end,:] .≈ data.metadata["momentum_flux"])
                @test all(interior(data["wT"])[1,1,end,:] .≈ data.metadata["temperature_flux"])

                @test all(interior(data["wu*"])[1,1,end,:] .≈ datasets.scalings.uw(data.metadata["momentum_flux"]))
                @test all(interior(data["wT*"])[1,1,end,:] .≈ datasets.scalings.wT(data.metadata["temperature_flux"]))
            end
        end
    end

    @testset "Scalings" begin
        for i in 1:length(Nzs_coarse)
            Nz_coarse = Nzs_coarse[i]
            datasets = multiple_datasets[i]
            
            us = vcat([interior(dataset["u*"]) for dataset in datasets.data]...)
            vs = vcat([interior(dataset["v*"]) for dataset in datasets.data]...)
            Ts = vcat([interior(dataset["T*"]) for dataset in datasets.data]...)

            @show length(us)

            uws = vcat([interior(dataset["wu*"]) for dataset in datasets.data]...)
            vws = vcat([interior(dataset["wv*"]) for dataset in datasets.data]...)
            wTs = vcat([interior(dataset["wT*"]) for dataset in datasets.data]...)

            @test isapprox(mean(us), 0f0, atol=eps(maximum(us)))
            @test isapprox(mean(vs), 0f0, atol=eps(maximum(vs)))
            @test isapprox(mean(Ts), 0f0, atol=eps(maximum(Ts)))

            @test isapprox(mean(uws), 0f0, atol=eps(maximum(us)))
            @test isapprox(mean(vws), 0f0, atol=eps(maximum(vs)))
            @test isapprox(mean(wTs), 0f0, atol=eps(maximum(Ts)))

            @test isapprox(std(us), 1f0, atol=eps(maximum(us)))
            @test isapprox(std(vs), 1f0, atol=eps(maximum(vs)))
            @test isapprox(std(Ts), 1f0, atol=eps(maximum(Ts)))

            @test isapprox(std(uws), 1f0, atol=eps(maximum(us)))
            @test isapprox(std(vws), 1f0, atol=eps(maximum(vs)))
            @test isapprox(std(wTs), 1f0, atol=eps(maximum(Ts)))
        end
    end

end