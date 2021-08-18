using WindMixing
using WindMixing: train_NDE
using WindMixing: write_metadata_NDE_training
using WindMixing: RiBasedDiffusivity
using Flux
using OrdinaryDiffEq
using WindMixing: load_data
using JLD2

@testset "NDE Training" begin
    train_files = [
      "constant_Qu_-5e-4_constant_Qb_3e-8_2",
      "diurnal_Qu_-5e-4_diurnal_Qb_3e-8_2",
    ]

    𝒟train = load_data(train_files)

    OUTPUT_PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output"
    FILE_NAME = "test_NDE_training"
    FILE_PATH = joinpath(OUTPUT_PATH, "$(FILE_NAME).jld2")
    EXTRACTED_FILE_PATH = joinpath(OUTPUT_PATH, "$(FILE_NAME)_extracted.jld2")


    ν₀ = 1f-4
    ν₋ = 1f-1
    ΔRi = 1f-1
    Riᶜ = 0.25f0
    Pr = 1f0

    N_inputs = 96
    hidden_units = 20
    N_outputs = 31

    weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, leakyrelu), Dense(hidden_units, N_outputs)))

    uw_NN = re(weights ./ 1f5)
    vw_NN = re(weights ./ 1f5)
    wT_NN = re(weights ./ 1f5)

    diffusivity_scheme = RiBasedDiffusivity(ν₀, ν₋, ΔRi, Riᶜ, Pr)
    training_fractions = (T=0.8f0, ∂T∂z=0.8f0, profile=0.5f0)

    train_parameters = Dict(
                               "ν₀" => ν₀, 
                               "ν₋" => ν₋, 
                              "ΔRi" => ΔRi, 
                              "Riᶜ" => Riᶜ, 
                               "Pr" => Pr, 
                                "κ" => 10f0,
    "modified_pacanowski_philander" => true, 
            "convective_adjustment" => false,
                   "train_gradient" => true,
                     "zero_weights" => true, 
                 "gradient_scaling" => 5f-3, 
               "training_fractions" => training_fractions,
                          "diurnal" => false,
               "diffusivity_scheme" => diffusivity_scheme
    )

    train_epoch = 1
    train_trange = 1:20:100
    train_iteration = 2
    train_optimizer = [ADAM(1e-4)]

    timestepper = ROCK4()

    write_metadata_NDE_training(FILE_PATH, train_files, train_epoch, train_trange, train_parameters, train_optimizer, uw_NN, vw_NN, wT_NN)
    uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, train_files, train_trange, timestepper, train_optimizer, train_epoch, FILE_PATH,
                                                         maxiters = train_iteration, 
                                               diffusivity_scheme = train_parameters["diffusivity_scheme"],
                                                   train_gradient = train_parameters["train_gradient"],
                                                     zero_weights = train_parameters["zero_weights"],
                                                #  gradient_scaling = train_parameters["gradient_scaling"],
                                               training_fractions = train_parameters["training_fractions"],
                                                          diurnal = train_parameters["diurnal"],
                                    )

    file = jldopen(FILE_PATH)
    @info file
    close(file)

    extract_NN(FILE_PATH, EXTRACTED_FILE_PATH, "NDE")

    extracted_file = jldopen(EXTRACTED_FILE_PATH)
    
    @info extracted_file

    close(extracted_file)

    rm(FILE_PATH)
    rm(EXTRACTED_FILE_PATH)
end