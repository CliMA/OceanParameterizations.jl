using OceanParameterizations
using WindMixing
using Flux
using OrdinaryDiffEq
using Plots
using ArgParse

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--reconstruct_fluxes"
            help = ""
            default = false
            arg_type = Bool

        "--enforce_surface_fluxes"
            help = ""
            default = true
            arg_type = Bool

        "--train_test_same"
            help = ""
            default = false
            arg_type = Bool

        "--subsample_frequency"
            help = ""
            default = 1
            arg_type = Int
    end

    return parse_args(settings)
end

@info "Parsing command line arguments..."
args = parse_command_line_arguments()

reconstruct_fluxes = args["reconstruct_fluxes"]
println("Reconstruct fluxes? $(reconstruct_fluxes)")

enforce_surface_fluxes = args["enforce_surface_fluxes"]
println("Enforce surface fluxes? $(enforce_surface_fluxes)")

subsample_frequency = args["subsample_frequency"]
println("Subsample frequency for training... $(subsample_frequency)")

train_test_same = args["train_test_same"]
println("Train and test on the same file? $(train_test_same)")

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

files = files[1:2]
for i=1:length(files)

    if train_test_same
        # Train on only file i
        train_files=files[i]
    else
        # Train on all except file i
        train_files = files[1:end .!= i]
    end

    𝒟train = WindMixing.data(train_files,
                        scale_type=ZeroMeanUnitVarianceScaling,
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency,
                        enforce_surface_fluxes=enforce_surface_fluxes)
    # Test on file i
    test_file = files[i]
    𝒟test = WindMixing.data(test_file,
                        override_scalings=𝒟train.scalings, # use the scalings from the training data
                        reconstruct_fluxes=reconstruct_fluxes,
                        subsample_frequency=subsample_frequency,
                        enforce_surface_fluxes=enforce_surface_fluxes)
    les = read_les_output(test_file)

    output_gif_directory="GP/subsample_$(subsample_frequency)/reconstruct_$(reconstruct_fluxes)/enforce_surface_fluxes_$(enforce_surface_fluxes)/train_test_same_$(train_test_same)/test_$(test_file)"
    directory = pwd() * "/$(output_gif_directory)/"
    mkpath(directory)
    file = directory*"_output.txt"
    touch(file)
    o = open(file, "w")

    write(o, "= = = = = = = = = = = = = = = = = = = = = = = = \n")
    println("Test file 1 ($(test_file))")
    write(o, "Test file: $(test_file) \n")
    write(o, "Output will be written to: $(output_gif_directory) \n")

    ## Gaussian Process Regression

    # A. Find the kernel that minimizes the prediction error on the training data
    # * Sweeps over length-scale hyperparameter value in logγ_range
    # * Sweeps over covariance functions
    # logγ_range=-1.0:0.5:1.0 # sweep over length-scale hyperparameter
    # uncomment the next three lines to try this but just for testing the GPR use the basic get_kernel stuff below
    # uw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
    # vw_kernel = best_kernel(𝒟train.vw, logγ_range=logγ_range)
    # wT_kernel = best_kernel(𝒟train.wT, logγ_range=logγ_range)

    # OR set the kernel manually here (to save a bunch of time):
    # Result of the hyperparameter search - optimize_GP_kernels.jl
    if reconstruct_fluxes
        uw_kernel = get_kernel(2,0.4,0.0,euclidean_distance)
        vw_kernel = get_kernel(2,0.5,0.0,euclidean_distance)
        wT_kernel = get_kernel(2,1.3,0.0,euclidean_distance)
    else
        uw_kernel = get_kernel(2,0.4,0.0,euclidean_distance)
        vw_kernel = get_kernel(2,0.4,0.0,euclidean_distance)
        wT_kernel = get_kernel(2,1.2,0.0,euclidean_distance)
    end

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
    # myanimate(xs, name) = animate_prediction(xs, name, 𝒟test, test_file;
    #                         legend_labels=["GP(u,v,T)","Truth"], directory=output_gif_directory)
    # myanimate(uw_GP, "uw")
    # myanimate(vw_GP, "vw")
    # myanimate(wT_GP, "wT")

    t  = 𝒟test.t
    Nz = 32
    f  = les.f⁰
    H  = Float32(abs(𝒟test.uw.z[end] - 𝒟test.uw.z[1]))
    τ  = Float32(abs(t[:,1][end] - t[:,1][1]))
    u_scaling = 𝒟test.scalings["u"]
    v_scaling = 𝒟test.scalings["v"]
    T_scaling = 𝒟test.scalings["T"]
    uw_scaling = 𝒟test.scalings["uw"]
    vw_scaling = 𝒟test.scalings["vw"]
    wT_scaling = 𝒟test.scalings["wT"]
    get_μ_σ(name) = (𝒟test.scalings[name].μ, 𝒟test.scalings[name].σ)
    μ_u, σ_u = get_μ_σ("u")
    μ_v, σ_v = get_μ_σ("v")
    μ_T, σ_T = get_μ_σ("T")
    μ_uw, σ_uw = get_μ_σ("uw")
    μ_vw, σ_vw = get_μ_σ("vw")
    μ_wT, σ_wT = get_μ_σ("wT")
    D_cell = Float32.(Dᶜ(Nz, 1/Nz))

    A = - τ / H
    B = f * τ

    function NDE_nondimensional_flux(x, p, t)
        u = x[1:Nz]
        v = x[Nz+1:2*Nz]
        T = x[2*Nz+1:96]
        dx₁ = A .* σ_uw ./ σ_u .* D_cell * uw_GP_model(x) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
        dx₂ = A .* σ_vw ./ σ_v .* D_cell * vw_GP_model(x) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx₃ = A .* σ_wT ./ σ_T .* D_cell * wT_GP_model(x)
        return [dx₁; dx₂; dx₃]
    end

    function time_window(t, uvT, trange)
        return (Float32.(t[trange]), Float32.(uvT[:,trange]))
    end

    timesteps = 1:1:length(t) #1:5:100
    uvT₀ = Float32.(𝒟test.uvT_scaled[:,1])

    t_train, uvT_train = time_window(t, 𝒟test.uvT_scaled, timesteps)
    t_train = Float32.(t_train ./ τ)

    prob = ODEProblem(NDE_nondimensional_flux, uvT₀, (t_train[1], t_train[end]), saveat=t_train)
    sol = solve(prob, Tsit5())

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
