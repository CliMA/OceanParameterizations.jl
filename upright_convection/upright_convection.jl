using Dates
using Printf
using Gen
using OceanTurb
using JLD2
using Plots
using BSON

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

    return T, zC, t
end

T, zC, t = load_data("free_convection_profiles.jld2")

plot_LES_figure(T, zC, t)
animate_LES_solution(T, zC, t)

error("Stopping")

# Modified from: https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/ForwardMap/fm.jl

"""
Generative model for free convection.
"""
@gen function free_convection_model(ℂ, constants, N, L, Δt, times, T₀, FT, ∂T∂z)
    # Uniform priors on all four KPP parameters.
    CSL  ~ uniform(0, 1)
    CNL  ~ uniform(0, 8)
    Cb_T ~ uniform(0, 6)
    CKE  ~ uniform(0, 5)

    parameters = KPP.Parameters(CSL=CSL, CNL=CNL, Cb_T=Cb_T, CKE=CKE)

    model = KPP.Model(N=N, H=L, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.T.data[1:N] .= avg(T₀, N)

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

# Physical constants
ρ₀ = file["parameters/density"]
cₚ = file["parameters/specific_heat_capacity"]
f  = file["parameters/coriolis_parameter"]
α  = file["buoyancy/equation_of_state/α"]
β  = file["buoyancy/equation_of_state/β"]
g  = file["buoyancy/gravitational_acceleration"]

constants = Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

# OceanTurb parameters
N = 16
L = file["grid/Lz"]
Δt = 60

Q = parse(Float64, file["parameters/surface_cooling"])
FT = -Q / (ρ₀*cₚ)
∂T∂z = file["parameters/temperature_gradient"]

ℂ = (1.0, 1.0, 1.0, 1.0)
Nt = 600
times = t[1:Nt]
T₀ = T[1, :]

trace = Gen.simulate(free_convection_model, (ℂ, constants, N, L, Δt, times, T₀, FT, ∂T∂z))

choices = Gen.get_choices(trace)
@sprintf("Choices made: CSL=%.5f, CNL=%.5f, Cb_T=%.5f, CKE=%.5f",
         choices[:CSL], choices[:CNL], choices[:Cb_T], choices[:CKE])

KPP_solution, KPP_zC = trace.retval

anim = @animate for n=1:5:Nt
    title = @sprintf("Deepening mixed layer: %.2f days", t[n] / 86400)
    plot(T[n, :], zC, linewidth=2,
         xlim=(19, 20), ylim=(-100, 0), label="LES",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title=title, legend=:bottomright, show=false)

    plot!(KPP_solution[:, n], KPP_zC, linewidth=2, label="KPP")
end

gif(anim, "deepening_mixed_layer_random_KPP_parameters.gif", fps=15)

T_cs = T_coarse_grained = zeros(Nt, N)
for n in 1:Nt
    T_cs[n, :] .= avg(T[n, :], N)
end

@gen function kpp_proposal(trace)
    CSL  ~ normal(trace[:CSL],  0.1)
    CNL  ~ normal(trace[:CNL],  0.1)
    Cb_T ~ normal(trace[:Cb_T], 0.1)
    CKE  ~ normal(trace[:CKE],  0.1)
    return nothing
end

function do_inference(model, model_args, data; n_samples, max_iters=10n_samples)
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

    traces = []
    CSL_samples = zeros(n_samples)
    CNL_samples = zeros(n_samples)
    CbT_samples = zeros(n_samples)
    CKE_samples = zeros(n_samples)

    n_steps = 0
    n_accepted_steps = 0

    trace, _ = Gen.generate(model, model_args, observations)
    while n_accepted_steps < n_samples
        trace, accepted = Gen.metropolis_hastings(trace, kpp_proposal, (), observations=observations)
        if accepted
            n_accepted_steps = n_accepted_steps + 1
            push!(traces, trace)

            choices = Gen.get_choices(trace)
            CSL_samples[n_accepted_steps] = choices[:CSL]
            CNL_samples[n_accepted_steps] = choices[:CNL]
            CbT_samples[n_accepted_steps] = choices[:Cb_T]
            CKE_samples[n_accepted_steps] = choices[:CKE]
        end
        n_steps = n_steps + 1
        @show n_steps, n_accepted_steps
        n_steps >= max_iters && break
    end

    println("# of accepted steps: $n_accepted_steps")
    println("# of steps: $n_steps")
    println("Acceptence ratio: $(n_accepted_steps/n_steps)")

    return traces, CSL_samples, CNL_samples, CbT_samples, CKE_samples
end

function do_inference_one_sample(model, model_args, data; iters)
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
    for _ in 1:iters
        trace, accepted = Gen.metropolis_hastings(trace, kpp_proposal, (), observations=observations)
    end

    return trace
end

model_args = (ℂ, constants, N, L, Δt, times, T₀, FT, ∂T∂z)
traces, CSL, CNL, Cb_T, CKE = [], [], [], [], []

# for _ in 1:25
#     _traces, _CSL, _CNL, _Cb_T, _CKE =
#         do_inference(free_convection_model, model_args, T_coarse_grained, n_samples=100, max_iters=500)
#     append!(traces, _traces)
#     append!(CSL, _CSL)
#     append!(CNL, _CNL)
#     append!(Cb_T, _Cb_T)
#     append!(CKE, _CKE)
# end
#
# for L in [CSL, CNL, Cb_T, CKE]
#     filter!(x -> x != 0, L)
# end

samples = 10
for n in 1:samples
    @info "Sample $n/$samples"
    trace = do_inference_one_sample(free_convection_model, model_args, T_coarse_grained, iters=100)
    push!(traces, trace)

    choices = Gen.get_choices(trace)
    push!(CSL, choices[:CSL])
    push!(CNL, choices[:CNL])
    push!(Cb_T, choices[:Cb_T])
    push!(CKE, choices[:CKE])
end

bson("inferred_KPP_parameters.bson", Dict(:CSL => CSL, :CNL => CNL, :Cb_T => Cb_T, :CKE => CKE))
