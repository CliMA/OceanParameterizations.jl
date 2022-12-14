##
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
##
train_files = ["wind_-5e-4_cooling_3e-8_new"]

Nz = 32
𝒟test = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
constants = (f=1f-4, α=2f-4, g=9.80655f0, Nz=32, H=256)

BCs_unscaled = (uw=(top=𝒟test.uw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    vw=(top=𝒟test.vw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    wT=(top=𝒟test.wT.coarse[end, 1], bottom=𝒟test.wT.coarse[1, 1]))
ICs_unscaled = (u=𝒟test.u.coarse[:,1], v=𝒟test.v.coarse[:,1], T=𝒟test.T.coarse[:,1])

CSL = 0.1   # Surface layer fraction
Cτ = 0.4   # Von Karman constant
CNL = 6.33  # Non-local flux proportionality constant

Cstab = 2.0   # Stable buoyancy flux parameter for wind-driven turbulence
Cunst = 6.4   # Unstable buoyancy flux parameter for wind-driven turbulence

Cn = 1.0   # Exponent for effect of stable buoyancy forcing on wind mixing
Cmτ_U = 0.25  # Exponent for effect of unstable buoyancy forcing on wind mixing of U
Cmτ_T = 0.5   # Exponent for effect of unstable buoyancy forcing on wind mixing of T
Cmb_U = 1/3   # Exponent for the effect of wind on convective mixing of U
Cmb_T = 1/3   # Exponent for effect of wind on convective mixing of T

Cd_U = 0.5   # Wind mixing regime threshold for momentum
Cd_T = 2.5   # Wind mixing regime threshold for tracers

Cb_U = 0.599 # Buoyancy flux parameter for convective turbulence
Cb_T = 1.36  # Buoyancy flux parameter for convective turbulence

CRi = 0.3   # Critical bulk Richardson number
CKE = 4.32  # Unresolved turbulence parameter

KU₀ = 1e-6  # Interior viscosity for velocity
KT₀ = 1e-7  # Interior diffusivity for temperature
# KS₀ = 1e-9  # Interior diffusivity for salinity

Cτb_U = (Cτ / Cb_U)^(1/Cmb_U) * (1 + Cunst*Cd_U)^(Cmτ_U/Cmb_U) - Cd_U  # Wind stress parameter for convective turbulence
Cτb_T = (Cτ / Cb_T)^(1/Cmb_T) * (1 + Cunst*Cd_T)^(Cmτ_T/Cmb_T) - Cd_T  # Wind stress parameter for convective turbulence

CKE₀ = 1e-11 # Minimum unresolved turbulence kinetic energy

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

    t = 𝒟test.t[1:1:1153]
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
plot(𝒟test.T.coarse[:, end], 𝒟test.T.z, label="LES")
plot!(T[:, end], 𝒟test.T.z, label="KPP")
# plot(T₀, 𝒟test.T.z)

##
T_coarse_sampled = 𝒟test.T.coarse[:, 1:100:end]
dim_output = length(T_coarse_sampled)
var(𝒟test.T.coarse)

Γ = 1e-4 * I

noise_dist = MvNormal(zeros(dim_output), Γ)
y_true = T_coarse_sampled[:] .+ rand(noise_dist)

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

N_ensemble = 18
N_iterations = 10

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y_true, Γ, Inversion(); rng=rng)
##
for i in 1:N_iterations
    @info i
    params_i = get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat([kpp_model(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

final_parameters = mean(final_ensemble, dims=2)[:]
##

##