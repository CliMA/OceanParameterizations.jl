using OceanParameterizations
using WindMixing
using Plots

reconstruct_fluxes = true
println("Reconstruct fluxes? $(reconstruct_fluxes)")

enforce_surface_fluxes = true
println("Enforce surface fluxes? $(enforce_surface_fluxes)")

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

logγ_range = collect(-1:0.05:1.5)
n = length(logγ_range)

# Rows correspond to kernel functions, cols correspond to logγ values
errors_uw = zeros(4,n)
errors_vw = zeros(4,n)
errors_wT = zeros(4,n)

## Gaussian Process Regression
# Find the kernel that minimizes the prediction error on the training data
# * Sweeps over length-scale hyperparameter value in logγ_range

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis",
            "weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]
output_directory="GP/subsample_$(subsample_frequency)/reconstruct_$(reconstruct_fluxes)/enforce_surface_fluxes_$(enforce_surface_fluxes)"
# isdir(dirname(output_directory)) ||
mkpath(output_directory)

for i=1:length(files)
    # Train on all except file i
    train_files = files[1:end .!= i]
    𝒟train = data(train_files,
                        scale_type=ZeroMeanUnitVarianceScaling,
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency,
                        enforce_surface_fluxes=enforce_surface_fluxes)
    # Test on file i
    test_file = files[i]
    𝒟test = data(test_file,
                        override_scalings=𝒟train.scalings, # use the scalings from the training data
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency,
                        enforce_surface_fluxes=enforce_surface_fluxes)

    for k=1:4
        errors_uw[k,:] .+= error_per_gamma(𝒟train.uw, 𝒟test.uw, k; logγ_range=logγ_range)
        errors_vw[k,:] .+= error_per_gamma(𝒟train.vw, 𝒟test.vw, k; logγ_range=logγ_range)
        errors_wT[k,:] .+= error_per_gamma(𝒟train.wT, 𝒟test.wT, k; logγ_range=logγ_range)
    end
end

# total error across the 6 test files (where you train on all but that test file) at each log(γ) value
function hyperparameter_landscapes(errors)
    myplot(i) = plot(logγ_range, errors[i,:], linewidth=4, c=:purple, xlabel="log(γ)", ylabel="Error", legend=false, yscale=:log10)
    layout = @layout [a; b; c; d]
    p = plot(myplot(1), myplot(2), myplot(3), myplot(4), layout=layout)
end

# From top to bottom the rows in the plot correspond to:
# "Squared Exponential Kernel"
# "Matérn Kernel ν=1/2"
# "Matérn Kernel ν=3/2"
# "Matérn Kernel ν=5/2"
# From left to right the columns in the plot correspond to uw, vw, wT
p1 = hyperparameter_landscapes(errors_uw)
p2 = hyperparameter_landscapes(errors_vw)
p3 = hyperparameter_landscapes(errors_wT)
layout = @layout [a b c]
p = plot(p1, p2, p3, layout=layout)
png(output_directory*"/hyperparameter_landscapes_uw_vw_wT.png")

# Extract the optimal kernel function and log(γ) value from error matrix
k_logγ(errors) = (argmin(errors_uw)[1], logγ_range[argmin(errors_uw)[2]])
k_uw, logγ_uw = k_logγ(errors_uw)
k_vw, logγ_vw = k_logγ(errors_vw)
k_wT, logγ_wT = k_logγ(errors_wT)

# Create kernel objects
uw_kernel = get_kernel(k_uw, logγ_uw, 0.0, euclidean_distance)
vw_kernel = get_kernel(k_vw, logγ_vw, 0.0, euclidean_distance)
wT_kernel = get_kernel(k_wT, logγ_wT, 0.0, euclidean_distance)
println(uw_kernel)
println(vw_kernel)
println(wT_kernel)


i=1
train_files = files[1:end .!= i]

subsample_frequency
𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    reconstruct_fluxes=reconstruct_fluxes,
                    subsample_frequency=subsample_frequency,
                    enforce_surface_fluxes=enforce_surface_fluxes)
test_file = files[i]
𝒟test = data(test_file,
                    override_scalings=𝒟train.scalings, # use the scalings from the training data
                    reconstruct_fluxes=reconstruct_fluxes,
                    subsample_frequency=subsample_frequency,
                    enforce_surface_fluxes=enforce_surface_fluxes)

# Trained GP models
uw_GP_model = gp_model(𝒟train.uw, uw_kernel)
vw_GP_model = gp_model(𝒟train.vw, vw_kernel)
wT_GP_model = gp_model(𝒟train.wT, wT_kernel)

# GP predictions on test data
uw_GP = predict(𝒟test.uw, uw_GP_model)
vw_GP = predict(𝒟test.vw, vw_GP_model)
wT_GP = predict(𝒟test.wT, wT_GP_model)

# Compare GP predictions to truth
myanimate(xs, name) = animate_prediction(xs, name, 𝒟test, test_file;
                        filename=name*"_optimal_kernel_showing_free_convection", legend_labels=["GP(u,v,T)","Truth"], directory=output_directory)
myanimate(uw_GP, "uw")
myanimate(vw_GP, "vw")
myanimate(wT_GP, "wT")
