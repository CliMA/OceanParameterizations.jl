using Oceananigans:
    CPU, RegularCartesianGrid, ZFaceField, ComputedField, set!, compute!, interior,
    FluxBoundaryCondition, GradientBoundaryCondition, TracerBoundaryConditions, fill_halo_regions!,
    Forcing, IncompressibleModel, Simulation, run!

using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.OutputWriters: NetCDFOutputWriter, TimeInterval
using Oceananigans.AbstractOperations: @at, ∂z


function convective_adjustment!(model, Δt, K)
    Nz, Δz = model.grid.Nz, model.grid.Δz
    T = model.tracers.T

    ∂T∂z = ComputedField(@at (Cell, Cell, Cell) ∂z(T))
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

function oceananigans_convective_adjustment_nn(ds; nn_filepath)
    ρ₀ = 1027.0
    cₚ = 4000.0
    f  = ds.metadata[:coriolis_parameter]
    α  = ds.metadata[:thermal_expansion_coefficient]
    β  = 0.0
    g  = ds.metadata[:gravitational_acceleration]

    heat_flux = ds.metadata[:heat_flux]
    ∂T₀∂z = ds.metadata[:dθdz_deep]

    times = dims(ds[:T], Ti)
    stop_time = times[end]

    zf = dims(ds[:wT], ZDim)
    zc = dims(ds[:T], ZDim)
    Nz = length(zc)
    Lz = abs(zf[1])

    ## Grid setup

    topo = (Periodic, Periodic, Bounded)
    grid = RegularCartesianGrid(topology=topo, size=(1, 1, Nz), extent=(1, 1, Lz))

    ## Boundary conditions

    T_bc_top = FluxBoundaryCondition(heat_flux)
    T_bc_bottom = GradientBoundaryCondition(∂T₀∂z)
    T_bcs = TracerBoundaryConditions(grid, top=T_bc_top, bottom=T_bc_bottom)

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
        ∂z_wT_field = ComputedField(@at (Cell, Cell, Cell) ∂z(wT_field))
        compute!(∂z_wT_field)
        return interior(∂z_wT_field)[:]
    end

    enforce_fluxes(wT) = cat(0, wT, heat_flux, dims=1)

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

    model_convective_adjustment = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,))
    model_neural_network = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,), forcing=(T=T_forcing,))

    T₀ = reshape(Array(ds[:T][Ti=1]), size(grid)...)
    set!(model_convective_adjustment, T=T₀)
    set!(model_neural_network, T=T₀)

    ## Simulation setup

    K = 100  # convective adjustment diffusivity

    function progress_convective_adjustment(simulation)
        clock = simulation.model.clock
        @info "Convective adjustment: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        convective_adjustment!(simulation.model, simulation.Δt, K)
        return nothing
    end

    function progress_neural_network(simulation)
        model = simulation.model
        clock = simulation.model.clock

        @info "Neural network: iteration = $(clock.iteration), time = $(prettytime(clock.time))"

        T = interior(model.tracers.T)[:]
        ∂z_wT_NN .= neural_network_forcing(T)

        convective_adjustment!(model, simulation.Δt, K)

        return nothing
    end

    Δt = ds.metadata[:interval]
    simulation_convective_adjustment = Simulation(model_convective_adjustment, Δt=Δt, iteration_interval=1,
                                                  stop_time=stop_time, progress=progress_convective_adjustment)
    simulation_neural_network = Simulation(model_neural_network, Δt=Δt, iteration_interval=1,
                                           stop_time=stop_time, progress=progress_neural_network)

    ## Output writing

    w_CA, T_CA = model_convective_adjustment.velocities.w, model_convective_adjustment.tracers.T
    outputs_CA = (T = T_CA, wT = ComputedField(w_CA * T_CA))
    simulation_convective_adjustment.output_writers[:solution] =
        NetCDFOutputWriter(model_convective_adjustment, outputs_CA, schedule=TimeInterval(Δt),
                           filepath="oceananigans_convective_adjustment.nc", mode="c")

    w_NN, T_NN = model_neural_network.velocities.w, model_neural_network.tracers.T
    outputs_NN = (T = T_NN, wT = ComputedField(w_NN * T_NN))
    simulation_neural_network.output_writers[:solution] =
        NetCDFOutputWriter(model_neural_network, outputs_NN, schedule=TimeInterval(Δt),
                           filepath="oceananigans_neural_network.nc", mode="c")

    run!(simulation_convective_adjustment)
    run!(simulation_neural_network)

    ds_ca = NCDstack("oceananigans_convective_adjustment.nc")
    ds_nn = NCDstack("oceananigans_neural_network.nc")

    T_ca = dropdims(Array(ds_ca[:T]), dims=(1, 2))
    T_nn = dropdims(Array(ds_nn[:T]), dims=(1, 2))

    return T_ca, T_nn
end
