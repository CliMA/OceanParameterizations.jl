##
using Printf
using JLD2
using DataDeps
using OceanTurb
using CairoMakie

include("modified_pacalowski_philander_model.jl")
# include("modified_diffusivity_model.jl")

ds = jldopen(joinpath(pwd(), "Data", "three_layer_constant_fluxes_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128__statistics.jld2"))
OUTPUT_NAME = "modified_diffusivity_tstep60_wind_mixing_higher_diffusivity_test.mp4"
SIMULATION_NAME = "Wind Mixing test"

## Load LES grid information, boundary conditions, and initial conditions
Nz = ds["grid/Nz"]
Lz = ds["grid/Lz"]
u₀ = ds["timeseries/u/0"][:]
v₀ = ds["timeseries/v/0"][:]
T₀ = ds["timeseries/T/0"][:]
Fu = ds["boundary_conditions/u_top"]
Fθ = ds["boundary_conditions/θ_top"]
f₀ = ds["parameters/coriolis_parameter"]

## Construct OceanTurb models

# ΔRis = [0.5:0.1:1.0...]
ΔRis = [0.1, 1.]
constants = OceanTurb.Constants(Float64, f=f₀)
parameters = cat(PacanowskiPhilander.Parameters(), [ModifiedPacanowskiPhilanderParameters(ΔRi = ΔRi) for ΔRi in ΔRis]..., dims=1)
models = cat(PacanowskiPhilander.Model(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=parameters[1]), 
                [ModifiedPacanowskiPhilanderModel(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=parameters[i+1]) for i in 1:length(ΔRis)]..., dims=1)
model_names = cat("Oceananigans.jl LES", "Pacanowski-Philander", ["Modified Pacanowski-Philander ΔRi = $ΔRi" for ΔRi in ΔRis]..., dims=1)
# pp_parameters = PacanowskiPhilander.Parameters()
# mpp_parameters = ModifiedPacanowskiPhilanderParameters()
# pp_model = PacanowskiPhilander.Model(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=pp_parameters)
# mpp_model = ModifiedPacanowskiPhilanderModel(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=mpp_parameters)

for model in models
    model.bcs[1].top = OceanTurb.FluxBoundaryCondition(Fu)
    model.bcs[3].top = OceanTurb.FluxBoundaryCondition(Fθ)

    model.solution[1].data[1:Nz] .= u₀
    model.solution[2].data[1:Nz] .= v₀
    model.solution[3].data[1:Nz] .= T₀
end

Δt = 60.0
times = [ds["timeseries/t/$i"] for i in keys(ds["timeseries/t"])]
Nt = length(times)

U_solutions = [zeros(Nz, Nt) for i in 1:length(models)]
V_solutions = [zeros(Nz, Nt) for i in 1:length(models)]
T_solutions = [zeros(Nz, Nt) for i in 1:length(models)]

U′W′_solutions = [zeros(Nz+1, Nt) for i in 1:length(models)]
V′W′_solutions = [zeros(Nz+1, Nt) for i in 1:length(models)]
W′T′_solutions = [zeros(Nz+1, Nt) for i in 1:length(models)]

Ri_solutions = [zeros(Nz+1, Nt) for i in 1:length(models)]

function get_diffusive_flux(field_index, model)
    flux = FaceField(model.grid)
    field = model.solution[field_index]
    K = model.timestepper.eqn.K[field_index]
    for i in interiorindices(flux)
        @inbounds flux[i] = - K(model, i) * ∂z(field, i)
    end
    return flux
end

function get_richardson_number_profile(model)
    Ri = FaceField(model.grid)
    for i in interiorindices(Ri)
        @inbounds Ri[i] = local_richardson(model, i)
    end
    return Ri
end

for i in 1:length(models)
    @info "Model $i/$(length(models))"
    model = models[i]
    U_solution = U_solutions[i]
    V_solution = V_solutions[i]
    T_solution = T_solutions[i]
    U′W′_solution = U′W′_solutions[i]
    V′W′_solution = V′W′_solutions[i]
    W′T′_solution = W′T′_solutions[i]
    Ri_solution = Ri_solutions[i]
    
    for n in 1:Nt
        OceanTurb.run_until!(model, Δt, times[n])
        @info "Time = $(times[n])"

        U_solution[:, n] .= model.solution[1][1:Nz]
        V_solution[:, n] .= model.solution[2][1:Nz]
        T_solution[:, n] .= model.solution[3][1:Nz]

        U′W′_solution[:, n] .= get_diffusive_flux(1, model)[1:Nz+1]
        V′W′_solution[:, n] .= get_diffusive_flux(2, model)[1:Nz+1]
        W′T′_solution[:, n] .= get_diffusive_flux(3, model)[1:Nz+1]
        
        U′W′_solution[Nz+1, n] = Fu

        Ri_solution[:, n] = get_richardson_number_profile(model)[1:Nz+1]
    end
end

# Get rid of ∞ and super large values.
Ri_solutions = [clamp.(Ri_solution, -1, 2) for Ri_solution in Ri_solutions]

## Plot!
##
U_LES_solution = zeros(Nz, Nt)
V_LES_solution = zeros(Nz, Nt)
T_LES_solution = zeros(Nz, Nt)

U′W′_LES_solution = zeros(Nz+1, Nt)
V′W′_LES_solution = zeros(Nz+1, Nt)
W′T′_LES_solution = zeros(Nz+1, Nt)
Ri_LES_solution = zeros(Nz+1, Nt)

for (n, iter) in zip(1:length(times), keys(ds["timeseries/t"]))
    U_LES_solution[:, n] = ds["timeseries/u/$iter"]
    V_LES_solution[:, n] = ds["timeseries/v/$iter"]
    T_LES_solution[:, n] = ds["timeseries/T/$iter"]
    
    U′W′_LES_solution[:, n] = ds["timeseries/wu/$iter"]
    V′W′_LES_solution[:, n] = ds["timeseries/wv/$iter"]
    W′T′_LES_solution[:, n] = ds["timeseries/wT/$iter"]
end

U′W′_LES_solution[end, :] .= Fu
W′T′_LES_solution[end, :] .= Fθ


zc = models[1].grid.zc
zf = models[1].grid.zf

frame = Node(1)
plot_title = @lift @sprintf("Modified Diffusivity %s: time = %s", SIMULATION_NAME, prettytime(times[$frame]))

U_models = [@lift U_solution[:, $frame] for U_solution in U_solutions]
V_models = [@lift V_solution[:, $frame] for V_solution in V_solutions]
T_models = [@lift T_solution[:, $frame] for T_solution in T_solutions]

U′W′_models = [@lift U′W′_solution[:, $frame] for U′W′_solution in U′W′_solutions]
V′W′_models = [@lift V′W′_solution[:, $frame] for V′W′_solution in V′W′_solutions]
W′T′_models = [@lift W′T′_solution[:, $frame] for W′T′_solution in W′T′_solutions]
Ri_models = [@lift Ri_solution[:, $frame] for Ri_solution in Ri_solutions]

U_LES = @lift U_LES_solution[:, $frame]
V_LES = @lift V_LES_solution[:, $frame]
T_LES = @lift T_LES_solution[:, $frame]

U′W′_LES = @lift U′W′_LES_solution[:, $frame]
V′W′_LES = @lift V′W′_LES_solution[:, $frame]
W′T′_LES = @lift W′T′_LES_solution[:, $frame]

fig = Figure(resolution=(1920, 1080))

colors = ["dodgerblue2", "forestgreen", "darkorange", "navyblue", "hotpink2", "saddlebrown", "gray27", "lawngreen"]

ax_U = fig[1, 1] = Axis(fig, xlabel="U (m/s)", ylabel="z (m)")
U_lines = cat(lines!(ax_U, U_LES, zc, linewidth=3, color="crimson"), 
                [lines!(ax_U, U_models[i], zc, linewidth=3, color=colors[i]) for i in 1:length(U_models)]..., dims=1)
xlims!(ax_U, -1, 1)
ylims!(ax_U, -Lz, 0)

ax_V = fig[1, 2] = Axis(fig, xlabel="V (m/s)", ylabel="z (m)")
V_lines = cat(lines!(ax_V, V_LES, zc, linewidth=3, color="crimson"), 
                [lines!(ax_V, V_models[i], zc, linewidth=3, color=colors[i]) for i in 1:length(V_models)]..., dims=1)
xlims!(ax_V, -1, 1)
ylims!(ax_V, -Lz, 0)

ax_T = fig[1, 3] = Axis(fig, xlabel="T (°C)", ylabel="z (m)")
T_lines = cat(lines!(ax_T, T_LES, zc, linewidth=3, color="crimson"), 
                [lines!(ax_T, T_models[i], zc, linewidth=3, color=colors[i]) for i in 1:length(T_models)]..., dims=1)
ylims!(ax_T, -Lz, 0)

ax_U′W′ = fig[2, 1] = Axis(fig, xlabel="U′W′ (m²/s²)", ylabel="z (m)")
U′W′_lines = cat(lines!(ax_U′W′, U′W′_LES, zf, linewidth=3, color="crimson"),
                    [lines!(ax_U′W′, U′W′_models[i], zf, linewidth=3, color=colors[i]) for i in 1:length(U′W′_models)]..., dims = 1)
xlims!(ax_U′W′, extrema(U′W′_LES_solution))
ylims!(ax_U′W′, -Lz, 0)

ax_V′W′ = fig[2, 2] = Axis(fig, xlabel="V′W′ (m²/s²)", ylabel="z (m)")
V′W′_lines = cat(lines!(ax_V′W′, V′W′_LES, zf, linewidth=3, color="crimson"),
                    [lines!(ax_V′W′, V′W′_models[i], zf, linewidth=3, color=colors[i]) for i in 1:length(V′W′_models)]..., dims = 1)
xlims!(ax_V′W′, all(V′W′_LES_solution .== 0) ? (-1, 1) : extrema(V′W′_LES_solution))
ylims!(ax_V′W′, -Lz, 0)

ax_W′T′ = fig[2, 3] = Axis(fig, xlabel="W′T′ (m/s ⋅ °C)", ylabel="z (m)")
W′T′_lines = cat(lines!(ax_W′T′, W′T′_LES, zf, linewidth=3, color="crimson"),
                    [lines!(ax_W′T′, W′T′_models[i], zf, linewidth=3, color=colors[i]) for i in 1:length(W′T′_models)]..., dims = 1)
xlims!(ax_W′T′, extrema(W′T′_LES_solution))
ylims!(ax_W′T′, -Lz, 0)

ax_Ri = fig[2, 4] = Axis(fig, xlabel="Richardson number", ylabel="z (m)")
Ri_lines = [lines!(ax_Ri, Ri_models[i], zf, linewidth=3, color=colors[i]) for i in 1:length(Ri_models)]
xlims!(ax_Ri, extrema(Ri_solutions[end]))
ylims!(ax_Ri, -Lz, 0)

legend = fig[1, 4] = Legend(fig, U_lines, model_names)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
trim!(fig.layout)

record(fig, joinpath(pwd(), "Output/$OUTPUT_NAME"), 1:length(times), framerate=15) do n
    @info "Animating Pacanowski-Philander wind-mixing frame $n/$(length(times))..."
    frame[] = n
end
##