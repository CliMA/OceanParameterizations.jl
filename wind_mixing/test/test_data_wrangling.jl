using Base: Integer
using WindMixing: TrainingDatasets
using WindMixing: load_data

using Oceananigans
using Oceananigans.OutputReaders
using JLD2

using Statistics: mean, std
using Oceananigans.Fields: Field, location
using Oceananigans.OutputReaders
using OceanParameterizations

@testset "Data Wrangling" begin
    simulations = [
        "wind_-5e-4_cooling_3e-8",
        "wind_-5e-4_cooling_3e-8_cubic" 
    ]

    Nzs_coarse = [32, 64]
    # multiple_datasets = [TrainingDatasets(dirnames, Nz_coarse=Nz) for Nz in Nzs_coarse]
    multiple_datasets = [load_data(simulations, Nz_coarse=Nz) for Nz in Nzs_coarse]

    profiles_str = ["u*", "v*", "T*"]
    fluxes_str = ["wu*", "wv*", "wT*"]
    profile_gradients_str = ["∂u∂z*", "∂v∂z*", "∂T∂z*"]

    @testset "Data Type" begin
        for datasets in multiple_datasets
            for data in datasets.data
                for str in [profiles_str; fluxes_str; profile_gradients_str]
                    @test interior(data[str]) isa SubArray{Float32}
                    @test data[str].times isa Array{Float32}

                    grid = data[str].grid

                    for property in propertynames(grid)
                        value = getproperty(grid, property)
                        if isa(value, Number) && !isa(value, Integer)
                            @test value isa Float32
                        end
                    end
                end

                for (key, value) in data.metadata
                    if isa(value, Number) && !isa(value, Integer)
                        @test value isa Float32
                    end
                end
            end
        end
    end

    @testset "Coarse-Grained Grid Size" begin
        for i in 1:length(Nzs_coarse)
            Nz_coarse = Nzs_coarse[i]
            datasets = multiple_datasets[i]
            for data in datasets.data
                for str in profiles_str
                    @test length(interior(data[str])[1,1,:,1]) == Nz_coarse
                end

                for str in [fluxes_str; profile_gradients_str]
                    @test length(interior(data[str])[1,1,:,1]) == Nz_coarse + 1
                end

                @test data["u*"].grid.Nz == Nz_coarse
                @test data["v*"].grid.Nz == Nz_coarse
                @test data["T*"].grid.Nz == Nz_coarse

                @test data["wu*"].grid.Nz == Nz_coarse
                @test data["wv*"].grid.Nz == Nz_coarse
                @test data["wT*"].grid.Nz == Nz_coarse
                end
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

            uws = vcat([interior(dataset["wu*"]) for dataset in datasets.data]...)
            vws = vcat([interior(dataset["wv*"]) for dataset in datasets.data]...)
            wTs = vcat([interior(dataset["wT*"]) for dataset in datasets.data]...)

            @test all([datasets.scalings == dataset.metadata["scalings"] for dataset in datasets.data])

            @test isapprox(mean(us), 0f0, atol=eps(maximum(abs, us)))
            @test isapprox(mean(vs), 0f0, atol=eps(maximum(abs, vs)))
            @test isapprox(mean(Ts), 0f0, atol=12 * eps(maximum(abs, Ts)))

            @test isapprox(mean(uws), 0f0, atol=eps(maximum(abs, us)))
            @test isapprox(mean(vws), 0f0, atol=eps(maximum(abs, vs)))
            @test isapprox(mean(wTs), 0f0, atol=eps(maximum(abs, Ts)))

            @test isapprox(std(us), 1f0, atol=eps(maximum(abs, us)))
            @test isapprox(std(vs), 1f0, atol=eps(maximum(abs, vs)))
            @test isapprox(std(Ts), 1f0, atol=eps(maximum(abs, Ts)))

            @test isapprox(std(uws), 1f0, atol=eps(maximum(abs, us)))
            @test isapprox(std(vws), 1f0, atol=eps(maximum(abs, vs)))
            @test isapprox(std(wTs), 1f0, atol=eps(maximum(abs, Ts)))
        end
    end

end