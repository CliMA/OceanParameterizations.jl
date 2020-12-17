using OceanParameterizations
using WindMixing
using Flux
using OrdinaryDiffEq
using Plots

reconstruct_fluxes = false
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

## Pick training and test simulations

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis",
            "weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]

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
    les = read_les_output(test_file)

    output_gif_directory="GP/subsample_$(subsample_frequency)/reconstruct_$(reconstruct_fluxes)/enforce_surface_fluxes_$(enforce_surface_fluxes)/test_$(test_file)"
    directory = pwd() * "/$(output_gif_directory)/"
    mkpath(directory)
    file = directory*"_output.txt"
    touch(file)
    o = open(file, "w")

    write(o, "= = = = = = = = = = = = = = = = = = = = = = = = \n")
    write(o, "Test file: $(test_file) \n")
    write(o, "Output will be written to: $(output_gif_directory) \n")

    ## Gaussian Process Regression

    # A. Find the kernel that minimizes the prediction error on the training data
    # * Sweeps over length-scale hyperparameter value in logγ_range
    # * Sweeps over covariance functions
    logγ_range=-1.0:0.5:1.0 # sweep over length-scale hyperparameter
    # uncomment the next three lines to try this but just for testing the GPR use the basic get_kernel stuff below
    uw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
    vw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
    wT_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)

    # OR set the kernel manually here (to save a bunch of time):
    # uw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
    # vw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
    # wT_kernel = get_kernel(1,0.1,0.0,euclidean_distance)

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

    uvT₀      = 𝒟test.uvT_scaled[:,1]
    zF_coarse = 𝒟test.uw.z
    zC_coarse = 𝒟test.u.z
    t         = 𝒟test.t
    f⁰        = les.f⁰
    Nz        = 32

    ∂z(vec) = (vec[1:Nz] .- vec[2:Nz+1]) ./ diff(zF_coarse)
    function f(dx, x, p, t)
        u = x[1:Nz]
        v = x[Nz+1:2*Nz]
        dx[1:Nz] .= -∂z(uw_GP_model(x)) .+ f⁰ .* v
        dx[Nz+1:2*Nz] .= -∂z(vw_GP_model(x)) .- f⁰ .* u
        dx[2*Nz+1:end] .= -∂z(wT_GP_model(x))
    end

    prob = ODEProblem(f, uvT₀, (t[1],t[289]), saveat=t)
    sol = solve(prob, ROCK4())

    split_array(uvT) = (uvT[1:Nz,:], uvT[Nz+1:2*Nz,:], uvT[2*Nz+1:end,:])
    u_pred, v_pred, T_pred = split_array(sol)

    u_pair = (u_pred, 𝒟test.u.scaled)
    v_pair = (v_pred, 𝒟test.v.scaled)
    T_pair = (T_pred, 𝒟test.T.scaled)

    myanimate(xs, name) = animate_prediction(xs, name, 𝒟test, test_file;
                                legend_labels=["GP(u,v,T)","Truth"], directory=output_gif_directory)
    myanimate(u_pair, "u")
    myanimate(v_pair, "v")
    myanimate(T_pair, "T")

    write(o, "GP prediction error on u........ $(mse(u_pair)) \n")
    write(o, "GP prediction error on v........ $(mse(v_pair)) \n")
    write(o, "GP prediction error on T........ $(mse(T_pair)) \n")

    # Close output file
    close(o)
end


# tpoint = 100
# split_vector(uvT) = (uvT[1:Nz], uvT[Nz+1:2*Nz], uvT[2*Nz+1:end])
# u_pred, v_pred, T_pred = split_vector(sol[:,tpoint])
# p1 = plot(u_pred, zC_coarse, label="GP_DE")
# plot!(𝒟test.u.scaled[:,tpoint], zC_coarse, label="truth")
# p2 = plot(v_pred, zC_coarse, label="GP_DE")
# plot!(𝒟test.v.scaled[:,tpoint], zC_coarse, label="truth")
# p3 = plot(T_pred, zC_coarse, label="GP_DE")
# plot!(𝒟test.T.scaled[:,tpoint], zC_coarse, label="truth")
# layout = @layout [a b c]
# p = plot(p1, p2, p3, layout=layout)
# png(p, "hello.png")
