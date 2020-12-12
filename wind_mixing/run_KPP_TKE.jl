using OceanParameterizations
using WindMixing

## Pick training and test simulations

output_gif_directory = "Output1"

train_files = ["strong_wind", "free_convection"]
test_file = "strong_wind"

𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")
𝒟test = data(test_file,
                    override_scalings=𝒟train.scalings, # use the scalings from the training data
                    animate=false,
                    animate_dir="$(output_gif_directory)/Testing")
les = read_les_output(test_file)

## KPP Parameterization (no training)

# les = read_les_output(test_file)
parameters = KPP.Parameters() # default parameters
KPP_model = closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)
predictions = KPP_model()
T_KPP = (predictions, 𝒟test.T.coarse)
mse(T_KPP)
animate_prediction(T_KPP, "T", 𝒟test, test_file; legend_labels=["KPP(T)", "truth"], filename="T_KPP_$(test_file)", directory=output_gif_directory)

## TKE Parameterization (no training; use default parameters)

# les = read_les_output(test_file)
parameters = TKEMassFlux.TKEParameters() # default parameters
TKE_model = closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)()
predictions = TKE_model()
T_TKE = (predictions, 𝒟test.T.coarse)
mse(T_TKE)
animate_prediction(T_TKE, "T", 𝒟test, test_file; legend_labels=["TKE(T)", "truth"], filename="T_TKE_$(test_file)", directory=output_gif_directory)
