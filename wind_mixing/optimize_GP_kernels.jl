using OceanParameterizations
using WindMixing
using Flux
using OrdinaryDiffEq
using Plots

reconstruct_fluxes = false
println("Reconstruct fluxes? $(reconstruct_fluxes)")

subsample_frequency = 32
println("Subsample frequency for training... $(subsample_frequency)")

file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_coriolis" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

## Pick training and test simulations

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis",
            "weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]

for i=1:length(files)

    # Train on all except file i
    train_files = files[1:end .!= i]
    𝒟train = data(train_files,
                        scale_type=ZeroMeanUnitVarianceScaling,
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency)
    # Test on file i
    test_file = files[i]
    𝒟test = data(test_file,
                        override_scalings=𝒟train.scalings, # use the scalings from the training data
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency)
    les = read_les_output(test_file)

    output_gif_directory="GP/subsample_$(subsample_frequency)/reconstruct_$(reconstruct_fluxes)/test_$(test_file)"

    ## Gaussian Process Regression
    # A. Find the kernel that minimizes the prediction error on the training data
    # * Sweeps over length-scale hyperparameter value in logγ_range
    # * Sweeps over covariance functions
    logγ_range=-1.0:0.5:1.0 # sweep over length-scale hyperparameter
    # uncomment the next three lines to try this but just for testing the GPR use the basic get_kernel stuff below
    uw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
    vw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
    wT_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)

    # Report the kernels and their properties
    write(o, "Kernel for u'w'..... $(uw_kernel) \n")
    write(o, "Kernel for v'w'..... $(vw_kernel) \n")
    write(o, "Kernel for w'T'..... $(wT_kernel) \n")

    # Trained GP models
    uw_GP_model = gp_model(𝒟train.uw, uw_kernel)
    vw_GP_model = gp_model(𝒟train.vw, vw_kernel)
    wT_GP_model = gp_model(𝒟train.wT, wT_kernel)

    # GP predictions on test data
    uw_GP = predict(𝒟test.uw, uw_GP_model)
    vw_GP = predict(𝒟test.vw, vw_GP_model)
    wT_GP = predict(𝒟test.wT, wT_GP_model)

    # Report GP prediction error on the fluxes
    write(o, "GP prediction error on u'w'..... $(mse(uw_GP)) \n")
    write(o, "GP prediction error on v'w'..... $(mse(vw_GP)) \n")
    write(o, "GP prediction error on w'T'..... $(mse(wT_GP)) \n")

    # Compare GP predictions to truth
    myanimate(xs, name) = animate_prediction(xs, name, 𝒟test, test_file;
                            legend_labels=["GP(u,v,T)","Truth"], directory=output_gif_directory)
    myanimate(uw_GP, "uw")
    myanimate(vw_GP, "vw")
    myanimate(wT_GP, "wT")
end
