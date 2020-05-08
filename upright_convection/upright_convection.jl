using Dates
using Printf
using Gen
using OceanTurb
using JLD2
using Plots
using BSON
using ClimateSurrogates

# For quick headless plotting with Plots.jl (without warnings).
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

# Headless plotting with PyPlot
ENV["MPLBACKEND"] = "Agg"

import PyPlot
const plt = PyPlot

include("plot_LES.jl")

function load_data(filepath)
    file = jldopen(filepath)

    Is = keys(file["timeseries/t"])

    zC = file["grid/zC"]
    Nz = file["grid/Nz"]
    Lz = file["grid/Lz"]
    Nt = length(Is)

    t = zeros(Nt)
    T = zeros(Nt, Nz)

    for (i, I) in enumerate(Is)
        t[i] = file["timeseries/t/$I"]
        T[i, :] = file["timeseries/T/$I"][1, 1, 2:Nz+1]
    end

    # Physical constants
    ρ₀ = file["parameters/density"]
    cₚ = file["parameters/specific_heat_capacity"]
    f  = file["parameters/coriolis_parameter"]
    α  = file["buoyancy/equation_of_state/α"]
    β  = file["buoyancy/equation_of_state/β"]
    g  = file["buoyancy/gravitational_acceleration"]

    constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    Q = parse(Float64, file["parameters/surface_cooling"])
    FT = -Q / (ρ₀*cₚ)
    ∂T∂z = file["parameters/temperature_gradient"]

    return T, zC, t, Nz, Lz, constants, Q, FT, ∂T∂z
end

"""
Generative model for free convection.
Modified from: https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/ForwardMap/fm.jl
"""
@gen function free_convection_model(constants, N, L, Δt, times, T₀, FT, ∂T∂z)
    # Uniform priors on all four KPP parameters.
    CSL  ~ uniform(0, 1)
    CNL  ~ uniform(0, 8)
    Cb_T ~ uniform(0, 6)
    CKE  ~ uniform(0, 5)

    parameters = KPP.Parameters(CSL=CSL, CNL=CNL, Cb_T=Cb_T, CKE=CKE)

    model = KPP.Model(N=N, H=L, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.T.data[1:N] .= coarse_grain(T₀, N)

    # Set boundary conditions
    model.bcs.T.top = FluxBoundaryCondition(FT)
    model.bcs.T.bottom = GradientBoundaryCondition(∂T∂z)

    Nt = length(times)
    solution = zeros(N, Nt)

    # loop the model
    for n in 1:Nt
        run_until!(model, Δt, times[n])
        @. solution[:, n] = model.solution.T[1:N]
    end

    # Put prior distributions on the temperature at every
    # grid point at every time step with a tiny bit of noise.
    for n in 1:Nt, i in 1:N
        {(:T, i, n)} ~ normal(solution[i, n], 0.01)
    end

    return solution, model.grid.zc
end

@gen function kpp_proposal(trace)
    CSL  ~ normal(trace[:CSL],  0.1*trace[:CSL])
    CNL  ~ normal(trace[:CNL],  0.1*trace[:CNL])
    Cb_T ~ normal(trace[:Cb_T], 0.1*trace[:Cb_T])
    CKE  ~ normal(trace[:CKE],  0.1*trace[:CKE])
    return nothing
end

function do_inference(model, model_args, data; iters, verbose=true)
    # Create a choice map that maps model addresses (:T, i, n)
    # to observed data T[i, n]. We leave the four KPP parameters
    # (:CSL, :CNL, :Cb_T, :CKE) unconstrained, because we want them
    # to be inferred.
    observations = Gen.choicemap()

    Nt, N = size(data)
    for n in 1:Nt, i in 1:N
        observations[(:T, i, n)] = data[n, i]
    end

    KPP_parameters = Gen.select(:CSL, :CNL, :Cb_T, :CKE)
    trace, _ = Gen.generate(model, model_args, observations)
    accepts = 0
    for i in 1:iters
        trace, accepted = Gen.metropolis_hastings(trace, kpp_proposal, (), observations=observations)
        accepts += accepted
        if verbose
            @info "Iteration $i, acceptance ratio: " * @sprintf("%.4f", accepts/i)
        end
    end

    return trace
end

@gen function generate_gp_kernel()
    l ~ gamma(1, 5000)
    σ² ~ gamma(1, 2)
    kernel = SquaredExponential(l, σ²)
    return kernel
end

@gen function train_convection_gp(x_train, y_train)
    kernel ~ generate_gp_kernel()
    return GaussianProcess(x_train, y_train, kernel)
end

@gen function predict_convection_gp(x_train, y_train, x_test, y_test)
    gp ~ train_convection_gp(x_train, y_train)

    Nt, N = length(x_test), length(x_test[1])

    u = x_test[1]
    for n in 2:Nt
        u = predict(gp, [u])
        for i in 1:N
            {(:u, n, i)} ~ normal(u[i], 0.01)
        end
    end

    return nothing
end

function infer_gp_hyperparameters(x_train, y_train, x_test, y_test; iters, verbose=true)
    observations = Gen.choicemap()

    Nt, N = length(x_test), length(x_test[1])
    for n in 2:Nt, i in 1:N
        observations[(:u, n, i)] = x_test[n][i]
    end

    gp_hyperparameters = select(:gp => :kernel => :l, :gp => :kernel => :σ²)
    trace, _ = Gen.generate(predict_convection_gp, (x_train, y_train, x_test, y_test), observations)
    accepts = 0

    for i in 1:iters
        trace, accepted = metropolis_hastings(trace, gp_hyperparameters, observations=observations)
        accepts += accepted
        if verbose
            @info "Iteration $i, acceptance ratio: " * @sprintf("%.4f", accepts/i)
        end
    end

    return trace
end

function test_convection_gp(gp, x_train, y_train, x_test, y_test)
    u = x_train[1]
    Nt, N = length(x_train) + length(x_test), length(u)

    us = []
    push!(us, u)

    for n in 2:Nt
        u = predict(gp, [u])
        push!(us, u)
    end

    return us
end

function animate_convection_gp(T, us)
    Nt, N = size(T)

    anim = @animate for n=1:5:Nt
        p = plot(T[n, :], zC, linewidth=2,
             xlim=(19, 20), ylim=(-100, 0), label="LES",
             xlabel="Temperature (C)", ylabel="Depth (z)",
             dpi=200, show=false)

        plot!(p, us[n], zC_cs, linewidth=2, label="GP", legend=:bottomright)
    end

    mp4(anim, "deepening_mixed_layer_GP.mp4", fps=15)
end
