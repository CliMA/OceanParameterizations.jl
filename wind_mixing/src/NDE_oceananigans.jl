using Statistics
using LinearAlgebra

using JLD2
using FileIO

using Oceananigans
using OceanParameterizations
using WindMixing

using Flux: Chain
using Oceananigans.BuoyancyModels: BuoyancyField
using Oceananigans.OutputReaders: FieldDataset
using Oceananigans.Fields: FieldSlicer
using Oceanostics.FlowDiagnostics: richardson_number_ccf!

function modified_pacanowski_philander_diffusivity(model, constants, p; K=10)
    Nz = model.grid.Nz

    u = model.velocities.u
    v = model.velocities.v
    T = model.tracers.T

    ν₀ = p["ν₀"]
    ν₋ = p["ν₋"]
    ΔRi = p["ΔRi"]
    Riᶜ = p["Riᶜ"]
    Pr = p["Pr"]

    α, g = constants.α, constants.g

    b = BuoyancyField(model)

    Ri = KernelComputedField(Center, Center, Face, richardson_number_ccf!, model,
                             computed_dependencies=(u, v, b), parameters=(dUdz_bg=0, dVdz_bg=0, N2_bg=0))
    compute!(Ri)

    ν = zeros(Nz+1)

    for i in 2:Nz
        ν[i] = ν₀ + ν₋ * tanh_step((Ri[1, 1, i] - Riᶜ) / ΔRi)
    end

    return ν, ν ./ Pr
end

# Note: This assumes a Prandtl number of Pr = 1.
function modified_pacanowski_philander!(model, constants, Δt, p)
    Nz = model.grid.Nz
    Δz = model.grid.Δz

    u = model.velocities.u
    v = model.velocities.v
    T = model.tracers.T

    ν_velocities, ν_T = modified_pacanowski_philander_diffusivity(model, constants, p)

    lower_diagonal_velocities = [-Δt / Δz ^ 2 * ν_velocities[i]   for i in 2:Nz]
    upper_diagonal_velocities = [-Δt / Δz ^ 2 * ν_velocities[i+1] for i in 1:Nz-1]
    lower_diagonal_T = [-Δt / Δz ^ 2 * ν_T[i]   for i in 2:Nz]
    upper_diagonal_T = [-Δt / Δz ^ 2 * ν_T[i+1] for i in 1:Nz-1]

    diagonal_velocites = zeros(Nz)
    diagonal_T = zeros(Nz)
    for i in 1:Nz-1
        diagonal_velocites[i] = 1 + Δt / Δz ^ 2 * (ν_velocities[i] + ν_velocities[i+1])
        diagonal_T[i] = 1 + Δt / Δz ^ 2 * (ν_T[i] + ν_T[i+1])
    end
    diagonal_velocites[Nz] = 1 + Δt / Δz ^ 2 * ν_velocities[Nz]
    diagonal_T[Nz] = 1 + Δt / Δz ^ 2 * ν_T[Nz]

    𝓛_velocities = Tridiagonal(lower_diagonal_velocities, diagonal_velocites, upper_diagonal_velocities)
    𝓛_T = Tridiagonal(lower_diagonal_T, diagonal_T, upper_diagonal_T)

    u′ = 𝓛_velocities \ interior(u)[:]
    v′ = 𝓛_velocities \ interior(v)[:]
    T′ = 𝓛_T \ interior(T)[:]

    set!(model, u=reshape(u′, (1, 1, Nz)))
    set!(model, v=reshape(v′, (1, 1, Nz)))
    set!(model, T=reshape(T′, (1, 1, Nz)))

    return nothing
end

function oceananigans_modified_pacanowski_philander_nn(uw_NN, vw_NN, wT_NN, constants, BCs, scalings, diffusivity_params; 
            BASELINE_RESULTS_PATH, NN_RESULTS_PATH, stop_time=36000, Δt=60, diffusivity_model=modified_pacanowski_philander_diffusivity)
    ρ₀ = 1027.0
    cₚ = 4000.0
    β  = 0.0

    f, α, g, Nz, Lz = constants.f, constants.α, constants.g, constants.Nz, constants.Lz

    uw_flux, vw_flux, wT_flux = BCs.top.uw, BCs.top.vw, BCs.top.wT

    ∂u₀∂z, ∂v₀∂z = BCs.bottom.u, BCs.bottom.v

    ## Grid setup

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(1, 1, Nz), extent=(1, 1, Lz))

    T₀_les = constants.T₀
    T₀ = reshape(coarse_grain(T₀_les, 32, Center), size(grid)...)

    ## Boundary conditions

    u_bc_top = FluxBoundaryCondition(uw_flux)
    u_bc_bottom = GradientBoundaryCondition(∂u₀∂z)
    u_bcs = UVelocityBoundaryConditions(grid, top=u_bc_top, bottom=u_bc_bottom)

    v_bc_top = FluxBoundaryCondition(vw_flux)
    v_bc_bottom = GradientBoundaryCondition(∂v₀∂z)
    v_bcs = VVelocityBoundaryConditions(grid, top=v_bc_top, bottom=v_bc_bottom)

    T_bc_top = FluxBoundaryCondition(wT_flux)
    ∂T₀∂z = (T₀[2] - T₀[1]) / grid.Δz
    T_bc_bottom = GradientBoundaryCondition(∂T₀∂z)
    T_bcs = TracerBoundaryConditions(grid, top=T_bc_top, bottom=T_bc_bottom)

    ## Neural network forcing

    u_scaling = scalings.u
    v_scaling = scalings.v
    T_scaling = scalings.T

    uw_scaling = scalings.uw
    vw_scaling = scalings.vw
    wT_scaling = scalings.wT
    
    μ_u, σ_u, μ_v, σ_v, μ_T, σ_T = u_scaling.μ, u_scaling.σ, v_scaling.μ, v_scaling.σ, T_scaling.μ, T_scaling.σ
    μ_uw, σ_uw, μ_vw, σ_vw, μ_wT, σ_wT = uw_scaling.μ, uw_scaling.σ, vw_scaling.μ, vw_scaling.σ, wT_scaling.μ, wT_scaling.σ

    function diagnose_baseline_flux_uw(model)
        ν, _ = modified_pacanowski_philander_diffusivity(model, constants, diffusivity_params)
        ∂u∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.u))
        compute!(∂u∂z)

        uw = -ν .* interior(∂u∂z)[1, 1, :]
        uw[end] = uw_flux
        return uw
    end

    function diagnose_baseline_flux_vw(model)
        ν, _ = modified_pacanowski_philander_diffusivity(model, constants, diffusivity_params)
        ∂v∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.v))
        compute!(∂v∂z)

        vw = -ν .* interior(∂v∂z)[1, 1, :]
        vw[end] = vw_flux
        return vw
    end

    function diagnose_baseline_flux_wT(model)
        _, ν = modified_pacanowski_philander_diffusivity(model, constants, diffusivity_params)
        ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(model.tracers.T))
        compute!(∂T∂z)

        wT = -ν .* interior(∂T∂z)[1, 1, :]
        wT[end] = wT_flux
        return wT
    end

    function ∂z_uw(uw)
        uw_field = ZFaceField(CPU(), grid)
        set!(uw_field, reshape(uw, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(uw_field, CPU(), nothing, nothing)
        ∂uw∂z = ComputedField(@at (Center, Center, Center) ∂z(uw_field))
        compute!(∂uw∂z)
        return interior(∂uw∂z)[:]
    end

    function ∂z_vw(vw)
        vw_field = ZFaceField(CPU(), grid)
        set!(vw_field, reshape(vw, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(vw_field, CPU(), nothing, nothing)
        ∂vw∂z = ComputedField(@at (Center, Center, Center) ∂z(vw_field))
        compute!(∂vw∂z)
        return interior(∂vw∂z)[:]
    end

    function ∂z_wT(wT)
        wT_field = ZFaceField(CPU(), grid)
        set!(wT_field, reshape(wT, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(wT_field, CPU(), nothing, nothing)
        ∂wT∂z = ComputedField(@at (Center, Center, Center) ∂z(wT_field))
        compute!(∂wT∂z)
        return interior(∂wT∂z)[:]
    end

    enforce_fluxes_uw(uw) = cat(0, uw, uw_flux, dims=1)
    enforce_fluxes_vw(vw) = cat(0, vw, vw_flux, dims=1)
    enforce_fluxes_wT(wT) = cat(0, wT, wT_flux, dims=1)
    
    function diagnose_NN_flux_uw(model)
        u = u_scaling.(interior(model.velocities.u)[:])
        v = v_scaling.(interior(model.velocities.v)[:])
        T = T_scaling.(interior(model.tracers.T)[:])
        uvT = [u; v; T]

        ∂u∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.u))
        compute!(∂u∂z)

        uw = enforce_fluxes_uw(inv(uw_scaling).(uw_NN(uvT)) .- inv(uw_scaling)(0))

        ν, _ = diffusivity_model(model, constants, diffusivity_params)

        ν∂u∂z = ν .* interior(∂u∂z)[:]
        uw = uw .- ν∂u∂z
        return uw
    end

    function diagnose_NN_flux_vw(model)
        u = u_scaling.(interior(model.velocities.u)[:])
        v = v_scaling.(interior(model.velocities.v)[:])
        T = T_scaling.(interior(model.tracers.T)[:])
        uvT = [u; v; T]

        ∂v∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.v))
        compute!(∂v∂z)

        vw = enforce_fluxes_vw(inv(vw_scaling).(vw_NN(uvT)) .- inv(vw_scaling)(0))

        ν, _ = diffusivity_model(model, constants, diffusivity_params)

        ν∂v∂z = ν .* interior(∂v∂z)[:]

        vw = vw .- ν∂v∂z

        return vw
    end

    function diagnose_NN_flux_wT(model)
        u = u_scaling.(interior(model.velocities.u)[:])
        v = v_scaling.(interior(model.velocities.v)[:])
        T = T_scaling.(interior(model.tracers.T)[:])
        uvT = [u; v; T]

        ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(model.tracers.T))
        compute!(∂T∂z)

        wT = enforce_fluxes_wT(inv(wT_scaling).(wT_NN(uvT)) .- inv(wT_scaling)(0))

        _, ν = diffusivity_model(model, constants, diffusivity_params)

        ν∂T∂z = ν .* interior(∂T∂z)[:]

        wT = wT .- ν∂T∂z

        return wT
    end

    NN_uw_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        uw_NN,
        uw -> inv(uw_scaling).(uw),
        uw -> uw .- inv(uw_scaling).(0),
        enforce_fluxes_uw,
        ∂z_uw
    )

    NN_vw_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        vw_NN,
        vw -> inv(vw_scaling).(vw),
        vw -> vw .- inv(vw_scaling).(0),
        enforce_fluxes_vw,
        ∂z_vw
    )

    NN_wT_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        wT_NN,
        wT -> inv(wT_scaling).(wT),
        wT -> wT .- inv(wT_scaling).(0),
        enforce_fluxes_wT,
        ∂z_wT
    )

    ∂z_uw_NN = zeros(Nz)
    forcing_params_uw = (; ∂z_uw_NN)
    @inline neural_network_∂z_uw(i, j, k, grid, clock, model_fields, p) = - p.∂z_uw_NN[k]
    u_forcing = Forcing(neural_network_∂z_uw, discrete_form=true, parameters=forcing_params_uw)

    ∂z_vw_NN = zeros(Nz)
    forcing_params_vw = (; ∂z_vw_NN)
    @inline neural_network_∂z_vw(i, j, k, grid, clock, model_fields, p) = - p.∂z_vw_NN[k]
    v_forcing = Forcing(neural_network_∂z_vw, discrete_form=true, parameters=forcing_params_vw)

    ∂z_wT_NN = zeros(Nz)
    forcing_params_wT = (; ∂z_wT_NN)
    @inline neural_network_∂z_wT(i, j, k, grid, clock, model_fields, p) = - p.∂z_wT_NN[k]
    T_forcing = Forcing(neural_network_∂z_wT, discrete_form=true, parameters=forcing_params_wT)

    ## Model setup

    model_baseline = IncompressibleModel(
                       grid = grid,
                   coriolis = FPlane(f=f),
        boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                   buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

    )

    model_neural_network = IncompressibleModel(
                       grid = grid,
                   coriolis = FPlane(f=f),
        boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                    forcing = (u=u_forcing, v=v_forcing, T=T_forcing),
                   buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)
    )

    set!(model_baseline, T=T₀)
    set!(model_neural_network, T=T₀)

    ## Simulation setup

    function progress_baseline(simulation)
        clock = simulation.model.clock

        if clock.iteration % 100 == 0
            @info "Baseline: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        end

        modified_pacanowski_philander!(simulation.model, constants, simulation.Δt, diffusivity_params)
        return nothing
    end

    function progress_neural_network(simulation)
        model = simulation.model
        clock = simulation.model.clock

        if clock.iteration % 1000 == 0
            @info "Neural network: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        end

        u = interior(model.velocities.u)[:]
        v = interior(model.velocities.v)[:]
        T = interior(model.tracers.T)[:]

        uvT = (u=u, v=v, T=T)
        ∂z_uw_NN .=  NN_uw_forcing(uvT)
        ∂z_vw_NN .=  NN_vw_forcing(uvT)
        ∂z_wT_NN .=  NN_wT_forcing(uvT)

        modified_pacanowski_philander!(simulation.model, constants, simulation.Δt, diffusivity_params)

        return nothing
    end

    simulation_baseline = Simulation(model_baseline,
                        Δt = Δt,
        iteration_interval = 1,
                 stop_time = stop_time,
                  progress = progress_baseline
    )

    simulation_neural_network = Simulation(model_neural_network,
                    Δt = Δt,
    iteration_interval = 1,
             stop_time = stop_time,
              progress = progress_neural_network
    )

    ## Output writing
    outputs_baseline = (
        u = model_baseline.velocities.u,
        v = model_baseline.velocities.v,
        T = model_baseline.tracers.T,
        uw = model -> diagnose_baseline_flux_uw(model),
        vw = model -> diagnose_baseline_flux_vw(model),
        wT = model -> diagnose_baseline_flux_wT(model),
    )

    simulation_baseline.output_writers[:solution] =
        JLD2OutputWriter(model_baseline, outputs_baseline,
            schedule = TimeInterval(600),
              prefix = BASELINE_RESULTS_PATH,
               force = true,
        # field_slicer = FieldSlicer(with_halos=true)
        )

    outputs_NN = (
         u = model_neural_network.velocities.u,
         v = model_neural_network.velocities.v,
         T = model_neural_network.tracers.T,
        uw = model_neural_network -> diagnose_NN_flux_uw(model_neural_network),
        vw = model_neural_network -> diagnose_NN_flux_vw(model_neural_network),
        wT = model_neural_network -> diagnose_NN_flux_wT(model_neural_network),
    )

    simulation_neural_network.output_writers[:solution] =
        JLD2OutputWriter(model_neural_network, outputs_NN,
            schedule = TimeInterval(600),
              prefix = NN_RESULTS_PATH,
               force = true,
        # field_slicer = FieldSlicer(with_halos=true)
        )    

    @info "Running baseline simulation..."
    run!(simulation_baseline)

    jldopen("$(BASELINE_RESULTS_PATH).jld2", "a") do file
        file["training_info/parameters"] = diffusivity_params
    end

    @info "Running modified pacanowski philander simulation + neural network..."
    run!(simulation_neural_network)

    jldopen("$(NN_RESULTS_PATH).jld2", "a") do file
        file["training_info/parameters"] = diffusivity_params
    end

    # ds_baseline = FieldDataset(joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl", "oceananigans_baseline.jld2"))
    # ds_nn = FieldDataset(joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl", "oceananigans_modified_pacanowski_philander_NN.jld2"))

    # return ds_baseline, ds_nn
    return nothing
end
