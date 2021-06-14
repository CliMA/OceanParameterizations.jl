function column_model_1D_kpp(constants, BCs, ICs, times, parameters=OceanTurb.KPP.Parameters())
    f, α, g, Nz, H = constants.f, constants.α, constants.g, constants.Nz, constants.H
    ρ₀ = 1027.0
    cₚ = 4000.0
    β  = 0.0
    kpp_constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    model = OceanTurb.KPP.Model(N=Nz, H=H, stepper=:BackwardEuler, constants=kpp_constants, parameters=parameters)

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

    Nt = length(times)
    U = zeros(Nz, Nt)
    V = zeros(Nz, Nt)
    T = zeros(Nz, Nt)

    UW = zeros(Nz+1, Nt)
    VW = zeros(Nz+1, Nt)
    WT = zeros(Nz+1, Nt)

    # loop the model
    Δt = times[2] - times[1]
    for n in 1:Nt
        OceanTurb.run_until!(model, Δt, times[n])

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

    return (; U, V, T, UW, VW, WT)
end