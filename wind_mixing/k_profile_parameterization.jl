using OceanTurb

function column_model_1D_kpp(ds; parameters=OceanTurb.KPP.Parameters())
    ρ₀ = 1027.0
    cₚ = 4000.0
    f = ds["parameters/coriolis_parameter"]
    α = ds["parameters/thermal_expansion_coefficient"]
    g = ds["parameters/gravitational_acceleration"]
    β  = 0.0
    constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    zf = ds["grid/zF"]
    N = 128 #N = 32
    H = abs(zf[1])

    model = OceanTurb.KPP.Model(N=N, H=H, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.U.data[1:N] .= 0
    model.solution.V.data[1:N] .= 0
    model.solution.T.data[1:N] .= Array(ds["timeseries/T/0"][1, 1, :])

    UW_flux = ds["parameters/momentum_flux"]
    VW_flux = 0
    WT_flux = ds["parameters/temperature_flux"]

    ∂U₀∂z = ds["parameters/boundary_condition_u_bottom"]
    ∂V₀∂z = ds["parameters/boundary_condition_u_bottom"]
    ∂T₀∂z = ds["parameters/boundary_condition_θ_bottom"]

    model.bcs.U.top = OceanTurb.FluxBoundaryCondition(UW_flux)
    model.bcs.V.top = OceanTurb.FluxBoundaryCondition(VW_flux)
    model.bcs.T.top = OceanTurb.FluxBoundaryCondition(WT_flux)

    model.bcs.U.bottom = OceanTurb.GradientBoundaryCondition(∂U₀∂z)
    model.bcs.V.bottom = OceanTurb.GradientBoundaryCondition(∂V₀∂z)
    model.bcs.T.bottom = OceanTurb.GradientBoundaryCondition(∂T₀∂z)

    times = 1000
    Nt = length(times)
    U = zeros(N, Nt)
    V = zeros(N, Nt)
    T = zeros(N, Nt)

    UW = zeros(N+1, Nt)
    VW = zeros(N+1, Nt)
    WT = zeros(N+1, Nt)

    # loop the model
    Δt = ds["timeseries/t/$(keys(ds["timeseries/t"])[2])"] - ds["timeseries/t/$(keys(ds["timeseries/t"])[1])"]
    for n in 1:Nt
        OceanTurb.run_until!(model, Δt, times[n])

        U[:, n] .= model.solution.U[1:N]
        V[:, n] .= model.solution.V[1:N]
        T[:, n] .= model.solution.T[1:N]

        UW[:, n] .= OceanTurb.diffusive_flux(:U, model)[1:N+1]
        VW[:, n] .= OceanTurb.diffusive_flux(:V, model)[1:N+1]
        WT[:, n] .= OceanTurb.diffusive_flux(:T, model)[1:N+1] .+ OceanTurb.KPP.nonlocal_temperature_flux(model)[1:N+1]

        UW[N+1, n] = UW_flux
        VW[N+1, n] = VW_flux
        WT[N+1, n] = WT_flux
    end

    return (; U, V, T, UW, VW, WT)
end

ds = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")
results = column_model_1D_kpp(ds)
close(ds)

@info results