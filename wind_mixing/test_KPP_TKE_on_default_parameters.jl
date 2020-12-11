using OceanParameterizations
using OceanTurb, Plots
using Oceananigans.Grids: Cell, Face

include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")

output_gif_directory = "Output1"

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis", "weak_wind_strong_cooling",
          "strong_wind_weak_cooling", "strong_wind_weak_heating"]

for test_file in files
    𝒟test = data(test_file,
                        override_scalings=𝒟train.scalings, # use the scalings from the training data
                        animate=false,
                        animate_dir="$(output_gif_directory)/Testing")

    ## KPP Parameterization (no training)
    les = read_les_output(test_file)
    parameters = KPP.Parameters() # default parameters
    KPP_model = closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)
    predictions = KPP_model()
    T_KPP = (predictions, 𝒟test.T.coarse)
    mse(T_KPP)
    animate_prediction(T_KPP, "T", 𝒟test, test_file; legend_labels=["KPP(T)", "truth"], filename="T_KPP_$(test_file)", directory=output_gif_directory)

    ## TKE Parameterization (no training; use default parameters)
    les = read_les_output(test_file)
    parameters = TKEMassFlux.TKEParameters() # default parameters
    TKE_model = closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)
    predictions = TKE_model()
    T_TKE = (predictions, 𝒟test.T.coarse)
    mse(T_TKE)
    animate_prediction(T_TKE, "T", 𝒟test, test_file; legend_labels=["TKE(T)", "truth"], filename="T_TKE_$(test_file)", directory=output_gif_directory)
end
