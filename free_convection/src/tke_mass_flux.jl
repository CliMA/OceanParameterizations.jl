import OceanTurb

function free_convection_tke_mass_flux(ds; parameters=OceanTurb.TKEMassFlux.TKEParameters(), Δt=600)

    ρ₀ = 1027.0
    cₚ = 4000.0
    f  = ds.metadata["coriolis_parameter"]
    α  = ds.metadata["thermal_expansion_coefficient"]
    β  = 0.0
    g  = ds.metadata["gravitational_acceleration"]
    constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    T = ds["T"]
    wT = ds["wT"]
    N = Nz = size(T, 3)
    zc = znodes(T)
    zf = znodes(wT)
    H = abs(zf[1])
    Nt = size(T, 4)
    times = T.times

    model = OceanTurb.TKEMassFlux.Model(
                      grid = OceanTurb.UniformGrid(N=N, H=H),
                   stepper = :BackwardEuler,
        eddy_diffusivities = OceanTurb.TKEMassFlux.RiDependentDiffusivities(),
                 constants = constants
    )

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.T.data[1:N] .= interior(T)[1, 1, :, 1]

    # Set boundary conditions
    FT = ds.metadata["temperature_flux"]
    ∂T∂z = ds.metadata["dθdz_deep"]
    model.bcs.T.top = OceanTurb.FluxBoundaryCondition(FT)
    model.bcs.T.bottom = OceanTurb.GradientBoundaryCondition(∂T∂z)

    solution = zeros(N, Nt)
    flux = zeros(N+1, Nt)

    for n in 1:Nt
        OceanTurb.run_until!(model, Δt, times[n])

        solution[:, n] .= model.solution.T[1:N]

        flux[:, n] .= OceanTurb.diffusive_flux(:T, model)[1:N+1]
        flux[N+1, n] = FT
    end

    return (T=solution, wT=flux)
end
