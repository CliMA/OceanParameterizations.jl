##
using Plots
using JLD2
using FileIO
using Statistics
using OceanParameterizations
using WindMixing
using OceanTurb
using PairPlots
using EnsembleKalmanProcesses.ParameterDistributions
##
f_180 = load("Data/kpp_eki_180ensemble_1000iters_4sim.jld2")
f_150 = load("Data/kpp_eki_150ensemble_1000iters_4sim.jld2")
f_100 = load("Data/kpp_eki_100ensemble_1000iters_4sim.jld2")
f_80 = load("Data/kpp_eki_80ensemble_1000iters_4sim.jld2")
f_50 = load("Data/kpp_eki_50ensemble_1000iters_4sim.jld2")
f_36 = load("Data/kpp_eki_36ensemble_1000iters_4sim.jld2")

f_15 = load("Data/kpp_eki_15ensemble_1000iters_4sim.jld2")
f_15_loc = load("Data/kpp_eki_15ensemble_1000iters_4sim_loc_delta_test.jld2")

train_files = [
    "wind_-5e-4_cooling_3e-8_new",
    "wind_-5e-4_new",
    "wind_-2.5e-4_heating_-2.5e-8",
    "cooling_5e-8_new"
]

𝒟tests = [WindMixing.data(file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false) for file in train_files]
BCs_unscaled = [(uw=(top=data.uw.coarse[end, 1], bottom=data.uw.coarse[1, 1]), 
                 vw=(top=data.vw.coarse[end, 1], bottom=data.uw.coarse[1, 1]), 
                 wT=(top=data.wT.coarse[end, 1], bottom=data.wT.coarse[1, 1])) for data in 𝒟tests]

ICs_unscaled = [(u=data.u.coarse[:,1], v=data.v.coarse[:,1], T=data.T.coarse[:,1]) for data in 𝒟tests]
##
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
    return T
end

function RMS(a, b)
    return sqrt.(mean((a .- b) .^ 2))
end

function RMS_ensemble(file, 𝒟tests, BCs, ICs)
    params = [mean(particles, dims=2)[:] for particles in file["ensemble_parameters"]]
    rmss = [zeros(length(params)) for i in 1:length(𝒟tests)]

    for (i, data) in pairs(𝒟tests)
        @info i
        truth = data.T.scaled
        scaling = ZeroMeanUnitVarianceScaling(data.T.coarse)
        for j in 1:length(params)
            results =  scaling.(kpp_model(params[j], BCs[i], ICs[i]))
            rmss[i][j] = RMS(truth, results)
        end
    end
    return rmss
end

function RMS_ensemble_localizer(file, 𝒟tests, BCs, ICs)
    n_localizer = Int(length(file) / 2)
    rmss_localizers = []
    for k in 1:n_localizer
        @info k
        params = [mean(particles, dims=2)[:] for particles in file["ensemble_parameters_$(k)"]]
        rmss = [zeros(length(params)) for i in 1:length(𝒟tests)]

        for (i, data) in pairs(𝒟tests)
            @info i
            truth = data.T.scaled
            scaling = ZeroMeanUnitVarianceScaling(data.T.coarse)
            for j in 1:length(params)
                results =  scaling.(kpp_model(params[j], BCs[i], ICs[i]))
                rmss[i][j] = RMS(truth, results)
            end
        end
        push!(rmss_localizers, rmss)
    end
    return rmss_localizers
end

RMS_ensemble(file) = RMS_ensemble(file, 𝒟tests, BCs_unscaled, ICs_unscaled)
RMS_ensemble_localizer(file) = RMS_ensemble_localizer(file, 𝒟tests, BCs_unscaled, ICs_unscaled)

function RMS_mean_sim(rmss)
    rms_mean = zeros(length(rmss[1]))
    for i in 1:length(rms_mean)
        rms_mean[i] = mean([rmss[j][i] for j in 1:length(rmss)])
    end
    return rms_mean
end
##
params_final_180 = mean(f_180["final_ensemble"], dims=2)[:]

params_180 = [mean(p, dims=2)[:] for p in f_180["ensemble_parameters"]]

T_SWSC_final_180 = kpp_model(params_final_180, BCs_unscaled[1], ICs_unscaled[1])

rmss_180 = RMS_ensemble(f_180)

rmss_150 = RMS_ensemble(f_150)
rmss_100 = RMS_ensemble(f_100)
rmss_80 = RMS_ensemble(f_80)
rmss_50 = RMS_ensemble(f_50)
rmss_36 = RMS_ensemble(f_36)

rmss_15 = RMS_ensemble(f_15)
rmss_15_localizer = RMS_ensemble(f_15_loc)
##
zs = 𝒟tests[1].T.z

plot(𝒟tests[1].T.coarse[:, 500], zs, label="LES", legend=:topleft)
plot!(T_SWSC_final_180[:, 500], zs)
plot!(𝒟tests[1].T.coarse[:, end], zs, label="LES", legend=:topleft)
plot!(T_SWSC_final_180[:, end], zs)



##
plot(1:length(rmss_180[1]), rmss_180[1], yscale=:log10, xscale=:log10, label="Strong wind strong cooling")
plot!(1:length(rmss_180[1]), rmss_180[2], yscale=:log10, xscale=:log10, label="Strong wind")
plot!(1:length(rmss_180[1]), rmss_180[3], yscale=:log10, xscale=:log10, label="Weak wind strong heating")
plot!(1:length(rmss_180[1]), rmss_180[4], yscale=:log10, xscale=:log10, label="Strong cooling")

##
params_final_150 = mean(f_150["final_ensemble"], dims=2)[:]
T_SW_final_150 = kpp_model(params_final_150, BCs_unscaled[2], ICs_unscaled[2])

params_uncalibrated = [0.1, 0.4, 6.33, 2.0, 6.4, 1.0, 0.25, 0.5, 1/3, 1/3, 0.5, 2.5, 0.599, 1.36, 0.3, 4.32, 1e-6, 1e-7]
T_SW_uncalibrated = kpp_model(params_uncalibrated, BCs_unscaled[2], ICs_unscaled[2])

plot(𝒟tests[2].T.coarse[:, 576], zs, label="LES, 4 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_SW_uncalibrated[:, 576], zs, label="Uncalibrated KPP, 4 days", linestyle=:dash, linewidth=2)
plot!(T_SW_final_150[:, 576], zs, label="Calibrated KPP, 4 days", linewidth=2)

plot!(𝒟tests[2].T.coarse[:, end], zs, label="LES, 8 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_SW_uncalibrated[:, end], zs, label="Uncalibrated KPP, 8 days", linestyle=:dash, linewidth=2)
plot!(T_SW_final_150[:, end], zs, label="Calibrated KPP, 8 days", linewidth=2)

xlabel!("Temperature (°C)")
ylabel!("z (m)")
savefig("Data/LES_KPP_SW.pdf")
##
T_WWSH_final_150 = kpp_model(params_final_150, BCs_unscaled[3], ICs_unscaled[3])
T_WWSH_uncalibrated = kpp_model(params_uncalibrated, BCs_unscaled[3], ICs_unscaled[3])

plot(𝒟tests[3].T.coarse[:, 576], zs, label="LES, 4 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_WWSH_uncalibrated[:, 576], zs, label="Uncalibrated KPP, 4 days", linestyle=:dash, linewidth=2)
plot!(T_WWSH_final_150[:, 576], zs, label="Calibrated KPP, 4 days", linewidth=2)

plot!(𝒟tests[3].T.coarse[:, end], zs, label="LES, 8 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_WWSH_uncalibrated[:, end], zs, label="Uncalibrated KPP, 8 days", linestyle=:dash, linewidth=2)
plot!(T_WWSH_final_150[:, end], zs, label="Calibrated KPP, 8 days", linewidth=2)

xlabel!("Temperature (°C)")
ylabel!("z (m)")
savefig("Data/LES_KPP_WWSH.pdf")
##
T_SWSC_final_150 = kpp_model(params_final_150, BCs_unscaled[1], ICs_unscaled[1])
T_SWSC_uncalibrated = kpp_model(params_uncalibrated, BCs_unscaled[1], ICs_unscaled[1])

plot(𝒟tests[1].T.coarse[:, 576], zs, label="LES, 4 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_SWSC_uncalibrated[:, 576], zs, label="Uncalibrated KPP, 4 days", linestyle=:dash, linewidth=2)
plot!(T_SWSC_final_150[:, 576], zs, label="Calibrated KPP, 4 days", linewidth=2)

plot!(𝒟tests[1].T.coarse[:, end], zs, label="LES, 8 days", legend=:topleft, linewidth=5, alpha=0.5)
plot!(T_SWSC_uncalibrated[:, end], zs, label="Uncalibrated KPP, 8 days", linestyle=:dash, linewidth=2)
plot!(T_SWSC_final_150[:, end], zs, label="Calibrated KPP, 8 days", linewidth=2)

xlabel!("Temperature (°C)")
ylabel!("z (m)")
savefig("Data/LES_KPP_SWSC.pdf")
##
rmss = [rmss_180, rmss_150, rmss_100, rmss_80, rmss_50, rmss_36]

rms_mean = [RMS_mean_sim(r) for r in rmss]

RMS_mean_sim(rmss_180)
N_ensemble = [180, 150, 100, 80, 50, 36]
rms_SWSC = [rms[1][end] for rms in rmss]
rms_SW = [rms[2][end] for rms in rmss]
rms_WWSH = [rms[3][end] for rms in rmss]
rms_SC = [rms[4][end] for rms in rmss]

scatter(N_ensemble, rms_SWSC, yscale=:log10, xscale=:log10, label="Strong wind strong cooling", legend=:bottomleft)
scatter!(N_ensemble, rms_SW, yscale=:log10, xscale=:log10, label="Strong wind")
scatter!(N_ensemble, rms_WWSH, yscale=:log10, xscale=:log10, label="Weak wind strong heating")
scatter!(N_ensemble, rms_SC, yscale=:log10, xscale=:log10, label="Strong cooling")

##
rms_15_mean = RMS_mean_sim(rmss_15)
rms_15_localizer_mean = RMS_mean_sim(rmss_15_localizer)

##
plot(1:1001, rms_mean[1], yscale=:log10, xscale=:log10, label="180 particles", palette = :tab10)
plot!(1:1001, rms_mean[2], yscale=:log10, xscale=:log10, label="150 particles")
plot!(1:1001, rms_mean[3], yscale=:log10, xscale=:log10, label="100 particles")
plot!(1:1001, rms_mean[4], yscale=:log10, xscale=:log10, label="80 particles")
plot!(1:1001, rms_mean[5], yscale=:log10, xscale=:log10, label="50 particles")
plot!(1:1001, rms_mean[6], yscale=:log10, xscale=:log10, label="36 particles")
xlabel!("Iterations")
ylabel!("RMSE across all simulations")
savefig("rms_Nparticles.pdf")
##
plot(1:1001, rms_15_mean, yscale=:log10, xscale=:log10, label="No localizer")
plot!(1:1001, rms_15_localizer_mean, yscale=:log10, xscale=:log10, label="RBF")
xlabel!("Iterations")
ylabel!("RMSE across all simulations")
savefig("Data/rms_15particles.pdf")
##
CSL = f_150["final_ensemble"][1,:]   # Surface layer fraction
Ctau = f_150["final_ensemble"][2,:]   # Von Karman constant
CNL = f_150["final_ensemble"][3,:]  # Non-local flux proportionality constant

Cstab = f_150["final_ensemble"][4,:]   # Stable buoyancy flux parameter for wind-driven turbulence
Cunst = f_150["final_ensemble"][5,:]   # Unstable buoyancy flux parameter for wind-driven turbulence

Cn = f_150["final_ensemble"][6,:]   # Exponent for effect of stable buoyancy forcing on wind mixing
Cmtau_U = f_150["final_ensemble"][7,:]  # Exponent for effect of unstable buoyancy forcing on wind mixing of U
Cmtau_T = f_150["final_ensemble"][8,:]   # Exponent for effect of unstable buoyancy forcing on wind mixing of T
Cmb_U = f_150["final_ensemble"][9,:]   # Exponent for the effect of wind on convective mixing of U
Cmb_T = f_150["final_ensemble"][10,:]   # Exponent for effect of wind on convective mixing of T

Cd_U = f_150["final_ensemble"][11,:]   # Wind mixing regime threshold for momentum
Cd_T = f_150["final_ensemble"][12,:]   # Wind mixing regime threshold for tracers

Cb_U = f_150["final_ensemble"][13,:] # Buoyancy flux parameter for convective turbulence
Cb_T = f_150["final_ensemble"][14,:]  # Buoyancy flux parameter for convective turbulence

CRi = f_150["final_ensemble"][15,:]   # Critical bulk Richardson number
CKE = f_150["final_ensemble"][16,:]  # Unresolved turbulence parameter

KU = f_150["final_ensemble"][17,:]  # Interior viscosity for velocity
KT = f_150["final_ensemble"][18,:]  # Interior diffusivity for temperature
##
table = (;
    CSL,   # Surface layer fraction
    # Ctau,   # Von Karman constant
    # CNL,  # Non-local flux proportionality constant

    # Cstab,   # Stable buoyancy flux parameter for wind-driven turbulence
    Cunst,   # Unstable buoyancy flux parameter for wind-driven turbulence

    # Cn,   # Exponent for effect of stable buoyancy forcing on wind mixing
    Cmtau_U,  # Exponent for effect of unstable buoyancy forcing on wind mixing of U
    # Cmtau_T,   # Exponent for effect of unstable buoyancy forcing on wind mixing of T
    # Cmb_U,   # Exponent for the effect of wind on convective mixing of U
    Cmb_T,   # Exponent for effect of wind on convective mixing of T

    # Cd_U,   # Wind mixing regime threshold for momentum
    # Cd_T,   # Wind mixing regime threshold for tracers

    # Cb_U, # Buoyancy flux parameter for convective turbulence
    Cb_T,  # Buoyancy flux parameter for convective turbulence

    CRi,   # Critical bulk Richardson number
    CKE,  # Unresolved turbulence parameter

    # KU,  # Interior viscosity for velocity
    # KT,
 )

corner(table, filterscatter=false)
savefig("Data/pair_plots_selected.png")
savefig("Data/pair_plots_selected.pdf")

##
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

##