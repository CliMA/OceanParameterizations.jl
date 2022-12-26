##
@info "Loading packages"
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using WindMixing
using OceanParameterizations
using OceanTurb
using Statistics
using Plots
using Distributions
using LinearAlgebra
const EKP = EnsembleKalmanProcesses
using Random
using JLD2

##
@info "Loading data"
train_files = [
    "wind_-5e-4_cooling_3e-8_new",   
    "wind_-5e-4_cooling_2e-8_new",   
    "wind_-5e-4_cooling_1e-8_new",   
    "wind_-3.5e-4_cooling_3e-8_new", 
    "wind_-3.5e-4_cooling_2e-8_new", 
    "wind_-3.5e-4_cooling_1e-8_new", 
    "wind_-2e-4_cooling_3e-8_new",   
    "wind_-2e-4_cooling_2e-8_new",   
    "wind_-2e-4_cooling_1e-8_new",   
    "wind_-5e-4_heating_-3e-8_new",  
    "wind_-5e-4_heating_-2e-8_new",  
    "wind_-5e-4_heating_-1e-8_new",  
    "wind_-3.5e-4_heating_-3e-8_new",
    "wind_-3.5e-4_heating_-2e-8_new",
    "wind_-3.5e-4_heating_-1e-8_new",
    "wind_-2e-4_heating_-3e-8_new",  
    "wind_-2e-4_heating_-2e-8_new",  
    "wind_-2e-4_heating_-1e-8_new",  
]

N_ensemble = parse(Int64, ARGS[1])
N_iteration = parse(Int64, ARGS[2])
N_sims = length(train_files)

# N_ensemble = 180
# N_iteration = 5

FILE_PATH = "Data/kpp_eki_$(N_ensemble)ensemble_$(N_iteration)iters_$(N_sims)sim.jld2"

function kpp_model(parameters, BCs, ICs)
    CSL, Cτ, CNL, Cstab, Cunst, Cn, Cmτ_U, Cmτ_T, Cmb_U, Cmb_T, Cd_U, Cd_T, Cb_U, Cb_T, CRi, CKE, KU₀, KT₀ = parameters

    Cτb_U = (Cτ / Cb_U)^(1/Cmb_U) * (1 + Cunst*Cd_U)^(Cmτ_U/Cmb_U) - Cd_U
    Cτb_T = (Cτ / Cb_T)^(1/Cmb_T) * (1 + Cunst*Cd_T)^(Cmτ_T/Cmb_T) - Cd_T

    kpp_parameters = OceanTurb.KPP.Parameters(; CSL, Cτ, CNL, Cstab, Cunst, Cn, Cmτ_U, Cmτ_T, Cmb_U, Cmb_T, Cd_U, Cd_T, Cb_U, Cb_T, CRi, CKE, KU₀, KT₀, Cτb_U, Cτb_T)

    f = 1f-4
    α = 2f-4
    g = 9.80655f0
    Nz = 32
    H = 256
    ρ₀ = 1027.0
    cₚ = 4000.0
    β  = 0.0
    kpp_constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    model = OceanTurb.KPP.Model(N=Nz, H=H, stepper=:BackwardEuler, constants=kpp_constants, parameters=kpp_parameters)
    # model = OceanTurb.KPP.Model(N=Nz, H=H, stepper=:BackwardEuler, constants=kpp_constants)

    U₀ = ICs.u
    V₀ = ICs.v
    T₀ = ICs.T

    model.solution.U.data[1:Nz] .= U₀
    model.solution.V.data[1:Nz] .= V₀
    model.solution.T.data[1:Nz] .= T₀

    UW_flux = BCs.uw.top
    VW_flux = BCs.vw.top
    WT_flux = BCs.wT.top

    Δz = H / Nz

    ∂U₀∂z = (U₀[2] - U₀[1]) / Δz
    ∂V₀∂z = (V₀[2] - V₀[1]) / Δz
    ∂T₀∂z = (T₀[2] - T₀[1]) / Δz

    model.bcs.U.top = OceanTurb.FluxBoundaryCondition(UW_flux)
    model.bcs.V.top = OceanTurb.FluxBoundaryCondition(VW_flux)
    model.bcs.T.top = OceanTurb.FluxBoundaryCondition(WT_flux)

    model.bcs.U.bottom = OceanTurb.GradientBoundaryCondition(∂U₀∂z)
    model.bcs.V.bottom = OceanTurb.GradientBoundaryCondition(∂V₀∂z)
    model.bcs.T.bottom = OceanTurb.GradientBoundaryCondition(∂T₀∂z)

    # t = 𝒟test.t[1:1:1153]
    t = range(0, step=600, length=1153)
    Nt = length(t)
    # U = zeros(Nz, Nt)
    # V = zeros(Nz, Nt)
    T = zeros(Nz, Nt)

    # UW = zeros(Nz+1, Nt)
    # VW = zeros(Nz+1, Nt)
    WT = zeros(Nz+1, Nt)

    # loop the model
    Δt = t[2] - t[1]
    for n in 1:Nt
        # @info n
        OceanTurb.run_until!(model, Δt, t[n])
        
        # if !isa(WT_flux, Number)
        #     model.bcs.T.top.condition = WT_flux(model.clock.time)
        # end

        # U[:, n] .= model.solution.U[1:Nz]
        # V[:, n] .= model.solution.V[1:Nz]
        T[:, n] .= model.solution.T[1:Nz]

        # UW[:, n] .= OceanTurb.diffusive_flux(:U, model)[1:Nz+1]
        # VW[:, n] .= OceanTurb.diffusive_flux(:V, model)[1:Nz+1]
        WT[:, n] .= OceanTurb.diffusive_flux(:T, model)[1:Nz+1] .+ OceanTurb.KPP.nonlocal_temperature_flux(model)[1:Nz+1]

        # UW[Nz+1, n] = UW_flux
        # VW[Nz+1, n] = VW_flux

        WT[Nz+1, n] = WT_flux
    end
    return T[:, 1:100:end][:]
end

kpp_model(parameters) = kpp_model(parameters, BCs_unscaled, ICs_unscaled)
##
function train_kpp_model(train_files, N_ensemble, N_iteration, FILE_PATH)
    Nz = 32
    𝒟tests = [WindMixing.data(file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false) for file in train_files]
    constants = (f=1f-4, α=2f-4, g=9.80655f0, Nz=32, H=256)
    
    BCs_unscaled = [(uw=(top=data.uw.coarse[end, 1], bottom=data.uw.coarse[1, 1]), 
        vw=(top=data.vw.coarse[end, 1], bottom=data.uw.coarse[1, 1]), 
        wT=(top=data.wT.coarse[end, 1], bottom=data.wT.coarse[1, 1])) for data in 𝒟tests]

    ICs_unscaled = [(u=data.u.coarse[:,1], v=data.v.coarse[:,1], T=data.T.coarse[:,1]) for data in 𝒟tests]

    T_coarse_sampled = vcat([data.T.coarse[:, 1:100:end][:] for data in 𝒟tests]...)

    function G(parameters)
        Ts = vcat([kpp_model(parameters, BCs_unscaled[i], ICs_unscaled[i]) for i in eachindex(𝒟tests)]...)
        return Ts
    end

    dim_output = length(T_coarse_sampled)

    Γ = 1e-4 * I

    noise_dist = MvNormal(zeros(dim_output), Γ)
    y_true = T_coarse_sampled .+ rand(noise_dist)

    std_scale = 4

    CSL_prior = constrained_gaussian("CSL", 0.1, 0.1 / std_scale, 0, 1) # Surface layer fraction
    Cτ_prior = constrained_gaussian("Cτ", 0.4, 0.4 / std_scale, 0, Inf) # Von Karman constant
    CNL_prior = constrained_gaussian("CNL", 6.33, 6.33 / std_scale, 0, Inf) # Non-local flux proportionality constant

    Cstab_prior = constrained_gaussian("Cstab", 2.0, 2.0 / std_scale, 0, Inf) # Stable buoyancy flux parameter for wind-driven turbulence
    Cunst_prior = constrained_gaussian("Cunst", 6.4, 6.4 / std_scale, 0, Inf) # Unstable buoyancy flux parameter for wind-driven turbulence

    Cn_prior = constrained_gaussian("Cn", 1.0, 1.0 / std_scale, 0, Inf) # Exponent for effect of stable buoyancy forcing on wind mixing
    Cmτ_U_prior = constrained_gaussian("Cmτ_U", 0.25, 0.25 / std_scale, 0, Inf) # Exponent for effect of unstable buoyancy forcing on wind mixing of U
    Cmτ_T_prior = constrained_gaussian("Cmτ_T", 0.5, 0.5 / std_scale, 0, Inf) # Exponent for effect of unstable buoyancy forcing on wind mixing of T
    Cmb_U_prior = constrained_gaussian("Cmb_U", 1 / 3, 1 / 3 / std_scale, 0, Inf) # Exponent for the effect of wind on convective mixing of U
    Cmb_T_prior = constrained_gaussian("Cmb_T", 1 / 3, 1 / 3 / std_scale, 0, Inf)   # Exponent for effect of wind on convective mixing of T

    Cd_U_prior = constrained_gaussian("Cd_U", 0.5, 0.5 / std_scale, 0, Inf) # Wind mixing regime threshold for momentum
    Cd_T_prior = constrained_gaussian("Cd_T", 2.5, 2.5 / std_scale, 0, Inf) # Wind mixing regime threshold for tracers

    Cb_U_prior = constrained_gaussian("Cb_U", 0.599, 0.599 / std_scale, 0, Inf) # Buoyancy flux parameter for convective turbulence
    Cb_T_prior = constrained_gaussian("Cd_T", 1.36, 1.36 / std_scale, 0, Inf) # Buoyancy flux parameter for convective turbulence

    CRi_prior = constrained_gaussian("CRi", 0.3, 0.3 / std_scale, 0, Inf) # Critical bulk Richardson number
    CKE_prior = constrained_gaussian("CKE", 4.32, 4.32 / std_scale, 0, Inf)  # Unresolved turbulence parameter

    KU₀_prior = constrained_gaussian("KU₀", 1e-6, 1e-6 / std_scale, 0, Inf) # Interior viscosity for velocity
    KT₀_prior = constrained_gaussian("KT₀", 1e-7, 1e-7 / std_scale, 0, Inf) # Interior diffusivity for temperature

    CSL_prior = constrained_gaussian("CSL", 0.1, 0.1 / std_scale, 0, 1) # Surface layer fraction
    Cτ_prior = constrained_gaussian("Cτ", 0.4, 0.4 / std_scale, 0, Inf) # Von Karman constant
    CNL_prior = constrained_gaussian("CNL", 6.33, 6.33 / std_scale, 0, Inf) # Non-local flux proportionality constant

    Cstab_prior = constrained_gaussian("Cstab", 2.0, 2.0 / std_scale, 0, Inf) # Stable buoyancy flux parameter for wind-driven turbulence
    Cunst_prior = constrained_gaussian("Cunst", 6.4, 6.4 / std_scale, 0, Inf) # Unstable buoyancy flux parameter for wind-driven turbulence

    Cn_prior = constrained_gaussian("Cn", 1.0, 1.0 / std_scale, 0, Inf) # Exponent for effect of stable buoyancy forcing on wind mixing
    Cmτ_U_prior = constrained_gaussian("Cmτ_U", 0.25, 0.25 / std_scale, 0, Inf) # Exponent for effect of unstable buoyancy forcing on wind mixing of U
    Cmτ_T_prior = constrained_gaussian("Cmτ_T", 0.5, 0.5 / std_scale, 0, Inf) # Exponent for effect of unstable buoyancy forcing on wind mixing of T
    Cmb_U_prior = constrained_gaussian("Cmb_U", 1 / 3, 1 / 3 / std_scale, 0, Inf) # Exponent for the effect of wind on convective mixing of U
    Cmb_T_prior = constrained_gaussian("Cmb_T", 1 / 3, 1 / 3 / std_scale, 0, Inf)   # Exponent for effect of wind on convective mixing of T

    Cd_U_prior = constrained_gaussian("Cd_U", 0.5, 0.5 / std_scale, 0, Inf) # Wind mixing regime threshold for momentum
    Cd_T_prior = constrained_gaussian("Cd_T", 2.5, 2.5 / std_scale, 0, Inf) # Wind mixing regime threshold for tracers

    Cb_U_prior = constrained_gaussian("Cb_U", 0.599, 0.599 / std_scale, 0, Inf) # Buoyancy flux parameter for convective turbulence
    Cb_T_prior = constrained_gaussian("Cd_T", 1.36, 1.36 / std_scale, 0, Inf) # Buoyancy flux parameter for convective turbulence

    CRi_prior = constrained_gaussian("CRi", 0.3, 0.3 / std_scale, 0, Inf) # Critical bulk Richardson number
    CKE_prior = constrained_gaussian("CKE", 4.32, 4.32 / std_scale, 0, Inf)  # Unresolved turbulence parameter

    KU₀_prior = constrained_gaussian("KU₀", 1e-6, 1e-6 / std_scale, 0, Inf) # Interior viscosity for velocity
    KT₀_prior = constrained_gaussian("KT₀", 1e-7, 1e-7 / std_scale, 0, Inf) # Interior diffusivity for temperature

    prior = combine_distributions([
        CSL_prior,
        Cτ_prior,
        CNL_prior,
        Cstab_prior,
        Cunst_prior,
        Cn_prior,
        Cmτ_U_prior,
        Cmτ_T_prior,
        Cmb_U_prior,
        Cmb_T_prior,
        Cd_U_prior,
        Cd_T_prior,
        Cb_U_prior,
        Cb_T_prior,
        CRi_prior,
        CKE_prior,
        KU₀_prior,
        KT₀_prior
    ])

    rng_seed = 41
    rng = Random.MersenneTwister(rng_seed)

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

    ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y_true, Γ, Inversion(); rng=rng)

    ##
    for i in 1:N_iteration
        @info i
        params_i = get_ϕ_final(prior, ensemble_kalman_process)

        G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

        EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    end

    final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

    jldopen(FILE_PATH, "w") do file
        file["final_ensemble"] = final_ensemble
    end

    # final_parameters = mean(final_ensemble, dims=2)[:]
end
##
@info "Starting up training"
train_kpp_model(train_files, N_ensemble, N_iteration, FILE_PATH)
##
# jldopen("Data/final_ensemble.jld2", "w") do file
#     file["final_ensemble"] = final_ensemble
# end

# f = jldopen("Data/final_ensemble.jld2")
# f["final_ensemble"]
# close(f)
##