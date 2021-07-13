function convective_adjustment!(model, Δt, K)
    Nz, Δz = model.grid.Nz, model.grid.Δz
    T = model.tracers.T

    ∂T∂z = ComputedField(@at (Center, Center, Center) ∂z(T))
    compute!(∂T∂z)

    κ = zeros(Nz)
    for i in 1:Nz
        κ[i] = ∂T∂z[1, 1, i] < 0 ? K : 0
    end

    ld = [-Δt/Δz^2 * κ[i]   for i in 2:Nz]
    ud = [-Δt/Δz^2 * κ[i+1] for i in 1:Nz-1]

    d = zeros(Nz)
    for i in 1:Nz-1
        d[i] = 1 + Δt/Δz^2 * (κ[i] + κ[i+1])
    end
    d[Nz] = 1 + Δt/Δz^2 * κ[Nz]

    𝓛 = Tridiagonal(ld, d, ud)

    T′ = 𝓛 \ interior(T)[:]
    set!(model, T=reshape(T′, (1, 1, Nz)))

    return nothing
end

function diagnose_convective_adjustment_wT(model, K, heat_flux)
    grid = model.grid
    Nz, Δz = grid.Nz, grid.Δz
    T = model.tracers.T

    ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(T))
    compute!(∂T∂z)

    κ_∂T∂z = ZFaceField(grid)

    for k in 1:Nz+1
        κ = ∂T∂z[1, 1, k] < 0 ? K : 0
        κ_∂T∂z[1, 1, k] = - κ * ∂T∂z[1, 1, k]
    end

    # Super dirty hack since ∂T∂z[1, 1, Nz] value seems huge and I have no idea why...
    κ_∂T∂z[1, 1, Nz+1] = heat_flux
    κ_∂T∂z[1, 1, Nz] = (κ_∂T∂z[1, 1, Nz-1] + κ_∂T∂z[1, 1, Nz+1]) / 2

    return κ_∂T∂z
end

function base_model(ds; forcing=NamedTuple())
    ρ₀ = 1027.0
    cₚ = 4000.0
    f  = ds.metadata["coriolis_parameter"]
    α  = ds.metadata["thermal_expansion_coefficient"]
    β  = 0.0
    g  = ds.metadata["gravitational_acceleration"]

    heat_flux = ds.metadata["temperature_flux"]
    ∂T₀∂z = ds.metadata["dθdz_deep"]

    T = ds["T"]
    wT = ds["wT"]
    Nz = size(T, 3)
    zc = znodes(T)
    zf = znodes(wT)
    Lz = abs(zf[1])

    Nt = size(T, 4)
    times = T.times
    stop_time = times[end]

    ## Grid setup

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(1, 1, Nz), extent=(1, 1, Lz))

    ## Boundary conditions

    T_bc_top = FluxBoundaryCondition(heat_flux)
    T_bc_bottom = GradientBoundaryCondition(∂T₀∂z)
    T_bcs = TracerBoundaryConditions(grid, top=T_bc_top, bottom=T_bc_bottom)

    ## Model setup

    model = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,), forcing=forcing)

    T₀ = reshape(Array(interior(T)[1, 1, :, 1]), size(grid)...)
    set!(model, T=T₀)

    return model
end

function oceananigans_convective_adjustment(ds; output_dir, Δt=600, K=10)
    model = base_model(ds)

    ## Simulation setup

    function progress_convective_adjustment(simulation)
        clock = simulation.model.clock
        # @info "Convective adjustment: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        convective_adjustment!(simulation.model, simulation.Δt, K)
        return nothing
    end

    stop_time = ds["T"].times[end]

    simulation = Simulation(model, Δt=Δt,
        iteration_interval = 1,
                 stop_time = stop_time,
                  progress = progress_convective_adjustment)

    ## Output writing

    heat_flux = ds.metadata["temperature_flux"]

    filepath = joinpath(output_dir, "oceananigans_convective_adjustment.nc")

    outputs = (
        T  = model.tracers.T,
        wT = model -> diagnose_convective_adjustment_wT(model, K, heat_flux)
    )

    simulation.output_writers[:solution] =
        NetCDFOutputWriter(model, outputs,
            schedule = TimeInterval(Δt),
            filepath = filepath,
                mode = "c",
          dimensions = (wT=("xC", "yC", "zF"),))

    @info "Running convective adjustment simulation..."
    run!(simulation)

    ds_nc = NCDataset(filepath)

    T = dropdims(Array(ds_nc["T"]), dims=(1, 2))
    wT = dropdims(Array(ds_nc["wT"]), dims=(1, 2))

    close(ds_nc)
    rm(filepath)

    solution = (; T, wT)

    return solution
end

function oceananigans_convective_adjustment_with_neural_network(ds; output_dir, nn_filepath, Δt=600)

    T = ds["T"]
    wT = ds["wT"]
    Nz = size(T, 3)
    zc = znodes(T)
    zf = znodes(wT)
    Lz = abs(zf[1])
    stop_time = T.times[end]
    heat_flux = ds.metadata["temperature_flux"]

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(1, 1, Nz), extent=(1, 1, Lz))

    ## Neural network forcing

    final_nn = jldopen(nn_filepath, "r")
    neural_network = final_nn["neural_network"]
    T_scaling = final_nn["T_scaling"]
    wT_scaling = final_nn["wT_scaling"]
    close(final_nn)

    μ_T, σ_T = T_scaling.μ, T_scaling.σ
    μ_wT, σ_wT = wT_scaling.μ, wT_scaling.σ

    function ∂z_wT(wT)
        wT_field = ZFaceField(CPU(), grid)
        set!(wT_field, reshape(wT, (1, 1, Nz+1)))
        fill_halo_regions!(wT_field, CPU(), nothing, nothing)
        ∂z_wT_field = ComputedField(@at (Center, Center, Center) ∂z(wT_field))
        compute!(∂z_wT_field)
        return interior(∂z_wT_field)[:]
    end

    enforce_fluxes(wT) = cat(0, wT, heat_flux, dims=1)

    # convective adjustment diffusivity
    K = wT_scaling.σ / T_scaling.σ * stop_time / Lz * 10

    function diagnose_wT_NN(model)
        T = interior(model.tracers.T)[:]
        T = T_scaling.(T)
        wT_NN_interior = neural_network(T)
        wT_NN_interior = inv(wT_scaling).(wT_NN_interior)
        wT_NN = enforce_fluxes(wT_NN_interior)

        ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(model.tracers.T))
        compute!(∂T∂z)

        κ = zeros(Nz+1)
        for i in 1:Nz+1
            κ[i] = ∂T∂z[1, 1, i] < 0 ? K : 0
        end

        K∂T∂z = κ .* interior(∂T∂z)[:]

        return wT_NN .- K∂T∂z
    end

    neural_network_forcing = Chain(
        T -> T_scaling.(T),
        neural_network,
        wT -> inv(wT_scaling).(wT),
        enforce_fluxes,
        ∂z_wT
    )

    ## TODO: Benchmark NN performance.

    ∂z_wT_NN = zeros(Nz)
    forcing_params = (∂z_wT_NN=∂z_wT_NN,)
    @inline neural_network_∂z_wT(i, j, k, grid, clock, model_fields, p) = - p.∂z_wT_NN[k]
    T_forcing = Forcing(neural_network_∂z_wT, discrete_form=true, parameters=forcing_params)

    ## Model setup

    model = base_model(ds, forcing=(T=T_forcing,))

    ## Simulation setup

    function progress_neural_network(simulation)
        model = simulation.model
        clock = simulation.model.clock

        # @info "Neural network: iteration = $(clock.iteration), time = $(prettytime(clock.time))"

        T = interior(model.tracers.T)[:]
        ∂z_wT_NN .= neural_network_forcing(T)

        convective_adjustment!(model, simulation.Δt, K)

        return nothing
    end

    simulation = Simulation(model, Δt=Δt, iteration_interval=1,
                            stop_time=stop_time, progress=progress_neural_network)

    ## Output writing

    filepath_NN = joinpath(output_dir, "oceananigans_neural_network.nc")
    outputs_NN = (T  = model.tracers.T,
                  wT = diagnose_wT_NN)

    simulation.output_writers[:solution] =
        NetCDFOutputWriter(model, outputs_NN,
                           schedule = TimeInterval(Δt),
                           filepath = filepath_NN,
                           mode = "c",
                           dimensions = (wT=("zF",),))

    @info "Running convective adjustment simulation + neural network..."
    run!(simulation)

    ds_nn = NCDataset(filepath_NN)

    T_nn = dropdims(Array(ds_nn["T"]), dims=(1, 2))
    wT_nn = Array(ds_nn["wT"])

    close(ds_nn)
    rm(filepath_NN)

    neural_network_solution = (T=T_nn, wT=wT_nn)

    return neural_network_solution
end
