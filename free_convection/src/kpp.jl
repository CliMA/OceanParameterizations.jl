using OceanTurb: KPP, Constants, FluxBoundaryCondition, GradientBoundaryCondition, run_until!

function free_convection_kpp(ds; parameters=KPP.Parameters())

    ρ₀ = 1027.0
    cₚ = 4000.0
    f  = ds.metadata[:coriolis_parameter]
    α  = ds.metadata[:thermal_expansion_coefficient]
    β  = 0.0
    g  = ds.metadata[:gravitational_acceleration]
    constants = Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

    zf = dims(ds[:wT], ZDim)
    zc = dims(ds[:T], ZDim)
    N = length(zc)
    L = abs(zf[1])
    model = KPP.Model(N=N, H=L, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.T.data[1:N] .= ds[:T][Ti=1]

    # Set boundary conditions
    FT = ds.metadata[:heat_flux]
    ∂T∂z = ds.metadata[:dθdz_deep]
    model.bcs.T.top = FluxBoundaryCondition(FT)
    model.bcs.T.bottom = GradientBoundaryCondition(∂T∂z)

    times = dims(ds[:T], Ti)
    Nt = length(times)
    solution = zeros(N, Nt)

    # loop the model
    Δt = ds.metadata[:interval]
    for n in 1:Nt
        run_until!(model, Δt, times[n])
        @. solution[:, n] = model.solution.T[1:N]
    end

    return solution
end
