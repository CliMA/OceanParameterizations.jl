using OceanParameterizations
using WindMixing
using OceanTurb

file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_coriolis" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis",
            "weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]

for test_file in files[1:2]
    𝒟test = data(test_file,
                        override_scalings=𝒟train.scalings, # use the scalings from the training data
                        animate=false,
                        animate_dir=output_directory*"/Testing")
    les = read_les_output(test_file)

    output_gif_directory="KPP_TKE/test_$(test_file)"
    mkpath(output_gif_directory)
    file = output_gif_directory*"_output.txt"
    touch(file)
    o = open(file, "w")

    write(o, "= = = = = = = = = = = = = = = = = = = = = = = = \n")
    write(o, "Test file: $(test_file) \n")
    write(o, "Output will be written to: $(output_gif_directory) \n")

    ## KPP Parameterization (no training)

    # les = read_les_output(test_file)
    parameters = KPP.Parameters() # default parameters
    KPP_model = closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], les)
    predictions = KPP_model()
    T_KPP = (predictions, 𝒟test.T.coarse)
    write(o, "KPP prediction error on T........ $(mse(T_KPP)) \n")
    animate_prediction(T_KPP, "T", 𝒟test, test_file; legend_labels=["KPP(T)", "truth"], filename="T_KPP_$(test_file)", directory=output_directory)

    ## TKE Parameterization (no training; use default parameters)

    # les = read_les_output(test_file)
    parameters = TKEMassFlux.TKEParameters() # default parameters
    TKE_model = closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], les)
    predictions = TKE_model()
    T_TKE = (predictions, 𝒟test.T.coarse)
    write(o, "KPP prediction error on T........ $(mse(T_TKE)) \n")
    animate_prediction(T_TKE, "T", 𝒟test, test_file; legend_labels=["TKE(T)", "truth"], filename="T_TKE_$(test_file)", directory=output_directory)

    close(o)
end
