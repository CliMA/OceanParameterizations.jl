##
using JLD2
using FileIO
using OceanParameterizations
using WindMixing
using OceanTurb
using Statistics
##
f = load("Data/kpp_eki_uvT_180ensemble_1000iters_18sim.jld2")

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
    U = zeros(Nz, Nt)
    V = zeros(Nz, Nt)
    T = zeros(Nz, Nt)

    UW = zeros(Nz+1, Nt)
    VW = zeros(Nz+1, Nt)
    WT = zeros(Nz+1, Nt)

    # loop the model
    Δt = t[2] - t[1]
    for n in 1:Nt
        @info n
        OceanTurb.run_until!(model, Δt, t[n])
        
        # if !isa(WT_flux, Number)
        #     model.bcs.T.top.condition = WT_flux(model.clock.time)
        # end

        U[:, n] .= model.solution.U[1:Nz]
        V[:, n] .= model.solution.V[1:Nz]
        T[:, n] .= model.solution.T[1:Nz]

        UW[:, n] .= OceanTurb.diffusive_flux(:U, model)[1:Nz+1]
        VW[:, n] .= OceanTurb.diffusive_flux(:V, model)[1:Nz+1]
        WT[:, n] .= OceanTurb.diffusive_flux(:T, model)[1:Nz+1] .+ OceanTurb.KPP.nonlocal_temperature_flux(model)[1:Nz+1]

        UW[Nz+1, n] = UW_flux
        VW[Nz+1, n] = VW_flux
        WT[Nz+1, n] = WT_flux
    end
    return U, V, T, UW, VW, WT
end

##
params_final = mean(f["final_ensemble"], dims=2)[:]

U, V, T, UW, VW, WT = kpp_model(params_final, BCs_unscaled[1], ICs_unscaled[1])

for (i, train_file) in pairs(train_files)
    U, V, T, UW, VW, WT = kpp_model(params_final, BCs_unscaled[i], ICs_unscaled[i])

    jldopen("Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2", "a+") do file
        file["$(train_file)/u"] = U
        file["$(train_file)/v"] = V
        file["$(train_file)/T"] = T
        file["$(train_file)/uw"] = UW
        file["$(train_file)/vw"] = VW
        file["$(train_file)/wT"] = WT
    end
end

jldopen("Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2", "a+") do file
    file["params"] = params_final
end

##

f_final = jldopen("Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2")
close(f_final)