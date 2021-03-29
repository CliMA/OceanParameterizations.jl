using Oceananigans
using Oceananigans.Buoyancy: BuoyancyField
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.OutputWriters: NetCDFOutputWriter, TimeInterval
using Oceananigans.AbstractOperations: @at, ∂z
using Oceanostics.FlowDiagnostics: richardson_number_ccf!
using FileIO
using JLD2
using Flux: Chain
using OceanParameterizations
using WindMixing
using Statistics
using LinearAlgebra

function modified_pacanowski_philander_diffusivity(model, ds, p)
    u = model.velocities.u
    v = model.velocities.v
    T = model.tracers.T    
    # ∂u∂z = ComputedField(@at (Center, Center, Center) ∂z(u))
    # ∂v∂z = ComputedField(@at (Center, Center, Center) ∂z(v))
    # ∂T∂z = ComputedField(@at (Center, Center, Center) ∂z(T))
    # compute!(∂u∂z)
    # compute!(∂v∂z)
    # compute!(∂T∂z)

    ν₀ = p["ν₀"]
    ν₋ = p["ν₋"]
    ΔRi = p["ΔRi"]
    Riᶜ = p["Riᶜ"]
    Pr = p["Pr"]
    α = ds["parameters/thermal_expansion_coefficient"]
    g = ds["parameters/gravitational_acceleration"]

    b = BuoyancyField(model)

    Ri = KernelComputedField(Center, Center, Face, richardson_number_ccf!, model,
                               computed_dependencies=(u, v, b), parameters=(dUdz_bg=0, dVdz_bg=0, N2_bg=0))
    compute!(Ri)

    Ri_smooth = zeros(1,1,33)

    Nz = model.grid.Nz + 1

    stencil = ones(5) .* 0.2

    for i in 3:Nz-2
        Ri_smooth[1,1,i] = sum(stencil .* Ri[i-2:i+2])
    end

    ν = zeros(Nz)

    tanh_step(x) = (1 - tanh(x)) / 2

    for i in 1:Nz
        # ν[i] = ν₀ + ν₋ * tanh_step((Ri[1, 1, i] - Riᶜ) / ΔRi)
        ν[i] = ν₀ + ν₋ * tanh_step((Ri_smooth[1, 1, i] - Riᶜ) / ΔRi)
    end

    ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(model.tracers.T))
    compute!(∂T∂z)

    K = 10

    κ = zeros(Nz)
    for i in 1:Nz
        κ[i] = ∂T∂z[1, 1, i] < 0 ? K : 0
    end

    return ν, κ
end


function modified_pacanowski_philander!(model, ds, Δt, p)
    Nz, Δz = model.grid.Nz, model.grid.Δz
    u = model.velocities.u
    v = model.velocities.v
    T = model.tracers.T

    ν_velocities, ν_T = modified_pacanowski_philander_diffusivity(model, ds, p)

    ld_velocities = [-Δt/Δz^2 * ν_velocities[i]   for i in 2:Nz]
    ud_velocities = [-Δt/Δz^2 * ν_velocities[i+1] for i in 1:Nz-1]

    ld_T = [-Δt/Δz^2 * ν_T[i]   for i in 2:Nz]
    ud_T = [-Δt/Δz^2 * ν_T[i+1] for i in 1:Nz-1]

    d_velocities = zeros(Nz)
    for i in 1:Nz-1
        d_velocities[i] = 1 + Δt/Δz^2 * (ν_velocities[i] + ν_velocities[i+1])
    end
    d_velocities[Nz] = 1 + Δt/Δz^2 * ν_velocities[Nz]

    d_T = zeros(Nz)
    for i in 1:Nz-1
        d_T[i] = 1 + Δt/Δz^2 * (ν_T[i] + ν_T[i+1])
    end
    d_T[Nz] = 1 + Δt/Δz^2 * ν_T[Nz]

    𝓛_velocities = Tridiagonal(ld_velocities, d_velocities, ud_velocities)
    𝓛_T = Tridiagonal(ld_T, d_T, ud_T)

    u′ = 𝓛_velocities \ interior(u)[:]
    v′ = 𝓛_velocities \ interior(v)[:]
    T′ = 𝓛_T \ interior(T)[:]

    set!(model, u=reshape(u′, (1, 1, Nz)))
    set!(model, v=reshape(v′, (1, 1, Nz)))
    set!(model, T=reshape(T′, (1, 1, Nz)))

    return nothing
end

# ds = training data
function coarse_grain(Φ, n, ::Type{Center})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

function oceananigans_modified_pacanowski_philander_nn(ds; output_dir, NN_filepath, stop_time=36000, Δt=60, diffusivity_model=modified_pacanowski_philander_diffusivity)
    ρ₀ = 1027.0
    cₚ = 4000.0
    # f  = ds.metadata[:coriolis_parameter]
    # α  = ds.metadata[:thermal_expansion_coefficient]
    β  = 0.0
    # g  = ds.metadata[:gravitational_acceleration]

    f = ds["parameters/coriolis_parameter"]
    α = ds["parameters/thermal_expansion_coefficient"]
    g = ds["parameters/gravitational_acceleration"]

    uw_flux = ds["parameters/boundary_condition_u_top"]
    vw_flux = 0
    wT_flux = ds["parameters/boundary_condition_θ_top"]
    
    ∂u₀∂z = ds["parameters/boundary_condition_u_bottom"]
    ∂v₀∂z = ds["parameters/boundary_condition_u_bottom"]
    ∂T₀∂z = ds["parameters/boundary_condition_θ_bottom"]

    Nz = 32
    Lz = ds["grid/Lz"]

    ## Grid setup

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(1, 1, Nz), extent=(1, 1, Lz))

    ## Boundary conditions

    u_bc_top = FluxBoundaryCondition(uw_flux)
    u_bc_bottom = GradientBoundaryCondition(∂u₀∂z)
    u_bcs = UVelocityBoundaryConditions(grid, top=u_bc_top, bottom=u_bc_bottom)
    
    v_bc_top = FluxBoundaryCondition(vw_flux)
    v_bc_bottom = GradientBoundaryCondition(∂v₀∂z)
    v_bcs = VVelocityBoundaryConditions(grid, top=v_bc_top, bottom=v_bc_bottom)

    T_bc_top = FluxBoundaryCondition(wT_flux)
    T_bc_bottom = GradientBoundaryCondition(∂T₀∂z)
    T_bcs = TracerBoundaryConditions(grid, top=T_bc_top, bottom=T_bc_bottom)

    ## Neural network forcing

    NN_file = jldopen(NN_filepath, "r")
    uw_NN = NN_file["neural_network/uw"]
    vw_NN = NN_file["neural_network/vw"]
    wT_NN = NN_file["neural_network/wT"]

    u_scaling = NN_file["training_info/u_scaling"]
    v_scaling = NN_file["training_info/v_scaling"]
    T_scaling = NN_file["training_info/T_scaling"]

    uw_scaling = NN_file["training_info/uw_scaling"]
    vw_scaling = NN_file["training_info/vw_scaling"]
    wT_scaling = NN_file["training_info/wT_scaling"]
    
    # ν₀ = NN_file["training_info/parameters"]["ν₀"]
    # ν₋ = NN_file["training_info/parameters"]["ν₋"]
    # ΔRi = NN_file["training_info/parameters"]["ΔRi"]
    # Riᶜ = NN_file["training_info/parameters"]["Riᶜ"]
    # Pr = NN_file["training_info/parameters"]["Pr"]
    diffusivity_params = NN_file["training_info/diffusivity_parameters"]
    close(NN_file)

    μ_u, σ_u, μ_v, σ_v, μ_T, σ_T = u_scaling.μ, u_scaling.σ, v_scaling.μ, v_scaling.σ, T_scaling.μ, T_scaling.σ
    μ_uw, σ_uw, μ_vw, σ_vw, μ_wT, σ_wT = uw_scaling.μ, uw_scaling.σ, vw_scaling.μ, vw_scaling.σ, wT_scaling.μ, wT_scaling.σ

    function ∂z_uw(uw)
        uw_field = ZFaceField(CPU(), grid)
        set!(uw_field, reshape(uw, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(uw_field, CPU(), nothing, nothing)
        ∂z_uw_field = ComputedField(@at (Center, Center, Center) ∂z(uw_field))
        compute!(∂z_uw_field)
        return interior(∂z_uw_field)[:]
    end

    function ∂z_vw(vw)
        vw_field = ZFaceField(CPU(), grid)
        set!(vw_field, reshape(vw, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(vw_field, CPU(), nothing, nothing)
        ∂z_vw_field = ComputedField(@at (Center, Center, Center) ∂z(vw_field))
        compute!(∂z_vw_field)
        return interior(∂z_vw_field)[:]
    end

    function ∂z_wT(wT)
        wT_field = ZFaceField(CPU(), grid)
        set!(wT_field, reshape(wT, (1, 1, Nz+1)))
        Oceananigans.fill_halo_regions!(wT_field, CPU(), nothing, nothing)
        ∂z_wT_field = ComputedField(@at (Center, Center, Center) ∂z(wT_field))
        compute!(∂z_wT_field)
        return interior(∂z_wT_field)[:]
    end

    enforce_fluxes_uw(uw) = cat(0, uw, uw_flux, dims=1)
    enforce_fluxes_vw(vw) = cat(0, vw, vw_flux, dims=1)
    enforce_fluxes_wT(wT) = cat(0, wT, wT_flux, dims=1)

    function diagnose_NN(model)
        u = u_scaling.(interior(model.velocities.u)[:])
        v = v_scaling.(interior(model.velocities.v)[:])
        T = T_scaling.(interior(model.tracers.T)[:])

        uvT = [u; v; T]

        ∂u∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.u))
        ∂v∂z = ComputedField(@at (Center, Center, Face) ∂z(model.velocities.v))
        ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(model.tracers.T))
        compute!(∂u∂z)
        compute!(∂v∂z)
        compute!(∂T∂z)

        uw = enforce_fluxes_uw(inv(uw_scaling).(uw_NN(uvT)))
        vw = enforce_fluxes_vw(inv(vw_scaling).(vw_NN(uvT)))
        wT = enforce_fluxes_wT(inv(wT_scaling).(wT_NN(uvT)))

        ν_velocities, ν_T = diffusivity_model(model, ds, diffusivity_params)

        ν∂u∂z = ν_velocities .* interior(∂u∂z)[:]
        ν∂v∂z = ν_velocities .* interior(∂v∂z)[:]
        ν∂T∂z = ν_T .* interior(∂T∂z)[:]

        return (uw=uw .- ν∂u∂z, vw=vw .- ν∂v∂z, wT=wT .- ν∂T∂z)
    end

    NN_uw_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        uw_NN,
        uw -> inv(uw_scaling).(uw),
        enforce_fluxes_uw,
        ∂z_uw
    )

    NN_vw_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        vw_NN,
        vw -> inv(vw_scaling).(vw),
        enforce_fluxes_vw,
        ∂z_vw
    )

    NN_wT_forcing = Chain(
        uvT -> [u_scaling.(uvT.u); v_scaling.(uvT.v); T_scaling.(uvT.T)],
        wT_NN,
        wT -> inv(wT_scaling).(wT),
        enforce_fluxes_wT,
        ∂z_wT
    )

    ## TODO: Benchmark NN performance.

    ∂z_uw_NN = zeros(Nz)
    forcing_params_uw = (∂z_uw_NN=∂z_uw_NN,)
    @inline neural_network_∂z_uw(i, j, k, grid, clock, model_fields, p) = - p.∂z_uw_NN[k]
    u_forcing = Forcing(neural_network_∂z_uw, discrete_form=true, parameters=forcing_params_uw)
    
    ∂z_vw_NN = zeros(Nz)
    forcing_params_vw = (∂z_vw_NN=∂z_vw_NN,)
    @inline neural_network_∂z_vw(i, j, k, grid, clock, model_fields, p) = - p.∂z_vw_NN[k]
    v_forcing = Forcing(neural_network_∂z_vw, discrete_form=true, parameters=forcing_params_vw)

    ∂z_wT_NN = zeros(Nz)
    forcing_params_wT = (∂z_wT_NN=∂z_wT_NN,)
    @inline neural_network_∂z_wT(i, j, k, grid, clock, model_fields, p) = - p.∂z_wT_NN[k]
    T_forcing = Forcing(neural_network_∂z_wT, discrete_form=true, parameters=forcing_params_wT)

    ## Model setup

    model_baseline = IncompressibleModel(
        grid=grid,
        coriolis = FPlane(f=f),
        boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs)
    )

    model_neural_network = IncompressibleModel(
        grid=grid,
        coriolis = FPlane(f=f),
        boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs,),
        forcing=(u=u_forcing, v=v_forcing, T=T_forcing,)
    )

    T₀ = reshape(coarse_grain(Array(ds["timeseries/T/0"][1,1,:]), 32, Center), size(grid)...)
    set!(model_baseline, T=T₀)
    set!(model_neural_network, T=T₀)

    ## Simulation setup

    function progress_baseline(simulation)
        clock = simulation.model.clock
        @info "Baseline: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        modified_pacanowski_philander!(simulation.model, ds, simulation.Δt, diffusivity_params)
        # simulation.model.tracers.T[1, 1, 1] = 19.645138
        return nothing
    end

    function progress_neural_network(simulation)
        model = simulation.model
        clock = simulation.model.clock

        @info "Neural network: iteration = $(clock.iteration), time = $(prettytime(clock.time))"
        u = interior(model.velocities.u)[:]
        v = interior(model.velocities.v)[:]
        T = interior(model.tracers.T)[:]
        uvT = (u=u, v=v, T=T)
        ∂z_uw_NN .=  NN_uw_forcing(uvT)
        ∂z_vw_NN .=  NN_vw_forcing(uvT)
        ∂z_wT_NN .=  NN_wT_forcing(uvT)
        modified_pacanowski_philander!(simulation.model, ds, simulation.Δt, diffusivity_params)
        simulation.model.tracers.T[1, 1, 1] = 19.645138
        return nothing
    end

    # Δt = ds.metadata[:interval]
    Δt = 60
    simulation_baseline = Simulation(model_baseline, Δt=Δt, iteration_interval=1,
                                                  stop_time=stop_time, progress=progress_baseline)
    simulation_neural_network = Simulation(model_neural_network, Δt=Δt, iteration_interval=1,
                                           stop_time=stop_time, progress=progress_neural_network)

    ## Output writing

    outputs_baseline = (u  = model_baseline.velocities.u, 
                        v  = model_baseline.velocities.v, 
                        T  = model_baseline.tracers.T, 
                        uw = model_baseline.velocities.w, 
                        )

    simulation_baseline.output_writers[:solution] =
        JLD2OutputWriter(model_baseline, outputs_baseline,
                           schedule = TimeInterval(600),
                           dir=output_dir,
                           prefix="oceananigans_baseline",
                           force=true)

    outputs_NN = (u=model_neural_network.velocities.u, 
                    v=model_neural_network.velocities.v, 
                    T=model_neural_network.tracers.T, 
                    uw=model_neural_network -> diagnose_NN(model_neural_network).uw, 
                    vw=model_neural_network -> diagnose_NN(model_neural_network).vw, 
                    wT=model_neural_network -> diagnose_NN(model_neural_network).wT,  
                    )

    simulation_neural_network.output_writers[:solution] =
        JLD2OutputWriter(model_neural_network, outputs_NN,
                           schedule = TimeInterval(600),
                           dir=output_dir,
                           prefix="oceananigans_modified_pacanowski_philander_NN",
                           force=true)

    @info "Running baseline simulation..."
    run!(simulation_baseline)

    @info "Running modified pacanowski philander simulation + neural network..."
    run!(simulation_neural_network)

    # ds_baseline = NCDstack(filepath_CA)
    # ds_nn = NCDstack(filepath_NN)

    # T_ca = dropdims(Array(ds_ca[:T]), dims=(1, 2))
    # T_nn = dropdims(Array(ds_nn[:T]), dims=(1, 2))
    # wT_nn = Array(ds_nn[:wT])

    # convective_adjustment_solution = (T=T_ca, wT=nothing)
    # neural_network_solution = (T=T_nn, wT=wT_nn)

    # return convective_adjustment_solution, neural_network_solution
end

𝒟train = WindMixing.data(["-1e-3"], scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

FILE_PATH_NN = joinpath(pwd(), "extracted_training_output", "NDE_training_modified_pacanowski_philander_1sim_-1e-3_diffusivity_1e-1_Ri_1e-1_new_3_extracted.jld2")

file = jldopen(FILE_PATH_NN, "r")
minimum(file["losses"])

uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

train_parameters = file["training_info/parameters"]
train_parameters["ν₋"] = 0.5f0

OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl"

jldopen(joinpath(OUTPUT_PATH, "NDE_oceananigans_extracted_test_2.jld2"), "w") do file
    file["neural_network/uw"] = uw_NN
    file["neural_network/vw"] = vw_NN
    file["neural_network/wT"] = wT_NN
    file["training_info/diffusivity_parameters"] = train_parameters
    file["training_info/u_scaling"] = 𝒟train.scalings["u"]
    file["training_info/v_scaling"] = 𝒟train.scalings["v"]
    file["training_info/T_scaling"] = 𝒟train.scalings["T"]
    file["training_info/uw_scaling"] = 𝒟train.scalings["uw"]
    file["training_info/vw_scaling"] = 𝒟train.scalings["vw"]
    file["training_info/wT_scaling"] = 𝒟train.scalings["wT"]
    file["training_info/uw_top"] = 𝒟train.uw.coarse[end,1]
    file["training_info/vw_top"] = 𝒟train.vw.coarse[end,1]
    file["training_info/wT_top"] = 𝒟train.wT.coarse[end,1]
end

LES_PATH = joinpath(pwd(), "Data", "three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")
ds = jldopen(LES_PATH, "r")

oceananigans_modified_pacanowski_philander_nn(ds, output_dir=OUTPUT_PATH, NN_filepath=joinpath(OUTPUT_PATH, "NDE_oceananigans_extracted_test_2.jld2"), stop_time=36000)

baseline_sol = jldopen(joinpath(OUTPUT_PATH, "oceananigans_baseline.jld2"), "r")
NN_sol = jldopen(joinpath(OUTPUT_PATH, "oceananigans_modified_pacanowski_philander_NN.jld2"), "r")

plot_data = NDE_profile_oceananigans(baseline_sol, NN_sol)

animate_profiles(plot_data, joinpath(OUTPUT_PATH, "NDE_oceananigans_extracted_test"), dimensionless=false)

close(baseline_sol)
close(NN_sol)
