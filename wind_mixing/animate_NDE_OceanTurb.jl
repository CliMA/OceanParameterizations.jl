using Statistics
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing
using JLD2
using FileIO
using OceanTurb
using CairoMakie
include("modified_pacalowski_philander_model.jl")

PATH = pwd()

DATA_PATH = joinpath(PATH, "extracted_training_output", "NDE_training_modified_pacalowski_philander_1sim_-1e-3_2_extracted.jld2")
FILE_PATH = joinpath(pwd(), "Output", "profiles_fluxes_modified_pacalowski_philander_1sim_-1e-3_test")

file = jldopen(DATA_PATH, "r")

losses = file["losses"]

minimum(losses)
size = length(losses)

train_files = file["training_info/train_files"]

Plots.plot(1:1:size, losses, yscale=:log10)
Plots.xlabel!("Iteration")
Plots.ylabel!("Loss mse")

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

test_files = ["-1e-3"]
𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]


trange = 1:1:1153
plot_data = NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange, unscale=true, modified_pacalowski_philander=true)

ds = jldopen(joinpath(pwd(), "Data", "three_layer_constant_fluxes_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128__statistics.jld2"))

Nz = ds["grid/Nz"]
Lz = ds["grid/Lz"]
u₀ = ds["timeseries/u/0"][:]
v₀ = ds["timeseries/v/0"][:]
T₀ = ds["timeseries/T/0"][:]
Fu = ds["boundary_conditions/u_top"]
Fθ = ds["boundary_conditions/θ_top"]
f₀ = ds["parameters/coriolis_parameter"]

ΔRi = 1.
ν₀ = 1e-4
ν₋ = 1e-1
Riᶜ = 0.25
Pr = 1.


constants = OceanTurb.Constants(Float64, f=f₀)
parameter = ModifiedPacanowskiPhilanderParameters(Cν₀=ν₀, Cν₋=ν₋, Riᶜ=Riᶜ, ΔRi=ΔRi)
model = ModifiedPacanowskiPhilanderModel(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=parameter) 

model.bcs[1].top = OceanTurb.FluxBoundaryCondition(Fu)
model.bcs[3].top = OceanTurb.FluxBoundaryCondition(Fθ)

model.solution[1].data[1:Nz] .= u₀
model.solution[2].data[1:Nz] .= v₀
model.solution[3].data[1:Nz] .= T₀

Δt = 10.0
times = [ds["timeseries/t/$i"] for i in keys(ds["timeseries/t"])]
Nt = length(times)

U_solution = zeros(Nz, Nt)
V_solution = zeros(Nz, Nt)
T_solution = zeros(Nz, Nt)

U′W′_solution = zeros(Nz+1, Nt)
V′W′_solution = zeros(Nz+1, Nt)
W′T′_solution = zeros(Nz+1, Nt)

Ri_solution = zeros(Nz+1, Nt)

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

Ri_solution = clamp.(Ri_solution, -1, 2)

frame = Node(1)

truth_u = @lift plot_data["truth_u"][:,$frame]
truth_v = @lift plot_data["truth_v"][:,$frame]
truth_T = @lift plot_data["truth_T"][:,$frame]

test_u = @lift plot_data["test_u"][:,$frame]
test_v = @lift plot_data["test_v"][:,$frame]
test_T = @lift plot_data["test_T"][:,$frame]

truth_uw = @lift plot_data["truth_uw"][:,$frame]
truth_vw = @lift plot_data["truth_vw"][:,$frame]
truth_wT = @lift plot_data["truth_wT"][:,$frame]

test_uw = @lift plot_data["test_uw"][:,$frame]
test_vw = @lift plot_data["test_vw"][:,$frame]
test_wT = @lift plot_data["test_wT"][:,$frame]

truth_Ri = @lift clamp.(plot_data["truth_Ri"][:,$frame], -1, 2)
test_Ri = @lift clamp.(plot_data["test_Ri"][:,$frame], -1, 2)

u_mpp = @lift U_solution[:,$frame]
v_mpp = @lift V_solution[:,$frame]
T_mpp = @lift T_solution[:,$frame]

uw_mpp = @lift U′W′_solution[:,$frame]
vw_mpp = @lift V′W′_solution[:,$frame]
wT_mpp = @lift W′T′_solution[:,$frame]

Ri_mpp = @lift Ri_solution[:,$frame]

u_max = maximum([maximum(plot_data["truth_u"]), maximum(plot_data["test_u"]), maximum(U_solution)])
u_min = minimum([minimum(plot_data["truth_u"]), minimum(plot_data["test_u"]), minimum(U_solution)])

v_max = maximum([maximum(plot_data["truth_v"]), maximum(plot_data["test_v"]), maximum(V_solution)])
v_min = minimum([minimum(plot_data["truth_v"]), minimum(plot_data["test_v"]), minimum(V_solution)])

T_max = maximum([maximum(plot_data["truth_T"]), maximum(plot_data["test_T"]), maximum(T_solution)])
T_min = minimum([minimum(plot_data["truth_T"]), minimum(plot_data["test_T"]), minimum(T_solution)])

uw_max = maximum([maximum(plot_data["truth_uw"]), maximum(plot_data["test_uw"]), maximum(U′W′_solution)])
uw_min = minimum([minimum(plot_data["truth_uw"]), minimum(plot_data["test_uw"]), minimum(U′W′_solution)])

vw_max = maximum([maximum(plot_data["truth_vw"]), maximum(plot_data["test_vw"]), maximum(V′W′_solution)])
vw_min = minimum([minimum(plot_data["truth_vw"]), minimum(plot_data["test_vw"]), minimum(V′W′_solution)])

wT_max = maximum([maximum(plot_data["truth_wT"]), maximum(plot_data["test_wT"]), maximum(W′T′_solution)])
wT_min = minimum([minimum(plot_data["truth_wT"]), minimum(plot_data["test_wT"]), minimum(W′T′_solution)])

SIMULATION_NAME="Modified Pacalowski Wind-Mixing, Training Data"
plot_title = @lift "$SIMULATION_NAME: time = $(round(times[$frame]/86400, digits=2)) days"
fig = Figure(resolution=(1920, 1080))

colors=["navyblue", "hotpink2", "forestgreen"]

u_str = "u / m s⁻¹"
v_str = "v / m s⁻¹"
T_str = "T / °C"
uw_str = "uw / m² s⁻²"
vw_str = "vw / m² s⁻²"
wT_str = "wT / m s⁻¹ °C"

zc = plot_data["depth_profile"]
zf = plot_data["depth_flux"]
z_str = "z / m"

zc_mpp = model.grid.zc
zf_mpp = model.grid.zf

ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
u_lines = [lines!(ax_u, truth_u, zc, linewidth=3, color=colors[1]), lines!(ax_u, test_u, zc, linewidth=3, color=colors[2]), lines!(ax_u, u_mpp, zc_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_u, u_min, u_max)
CairoMakie.ylims!(ax_u, minimum(zc), 0)

ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
v_lines = [lines!(ax_v, truth_v, zc, linewidth=3, color=colors[1]), lines!(ax_v, test_v, zc, linewidth=3, color=colors[2]), lines!(ax_v, v_mpp, zc_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_v, v_min, v_max)
CairoMakie.ylims!(ax_v, minimum(zc), 0)

ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
T_lines = [lines!(ax_T, truth_T, zc, linewidth=3, color=colors[1]), lines!(ax_T, test_T, zc, linewidth=3, color=colors[2]), lines!(ax_T, T_mpp, zc_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_T, T_min, T_max)
CairoMakie.ylims!(ax_T, minimum(zc), 0)

ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
uw_lines = [lines!(ax_uw, truth_uw, zf, linewidth=3, color=colors[1]), lines!(ax_uw, test_uw, zf, linewidth=3, color=colors[2]), lines!(ax_uw, uw_mpp, zf_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_uw, uw_min, uw_max)
CairoMakie.ylims!(ax_uw, minimum(zf), 0)

ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
vw_lines = [lines!(ax_vw, truth_vw, zf, linewidth=3, color=colors[1]), lines!(ax_vw, test_vw, zf, linewidth=3, color=colors[2]), lines!(ax_vw, vw_mpp, zf_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_vw, vw_min, vw_max)
CairoMakie.ylims!(ax_vw, minimum(zf), 0)

ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
wT_lines = [lines!(ax_wT, truth_wT, zf, linewidth=3, color=colors[1]), lines!(ax_wT, test_wT, zf, linewidth=3, color=colors[2]), lines!(ax_wT, wT_mpp, zf_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_wT, wT_min, wT_max)
CairoMakie.ylims!(ax_wT, minimum(zf), 0)

ax_Ri = fig[2, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
Ri_lines = [lines!(ax_Ri, truth_Ri, zf, linewidth=3, color=colors[1]), lines!(ax_Ri, test_Ri, zf, linewidth=3, color=colors[2]), lines!(ax_Ri, Ri_mpp, zf_mpp, linewidth=3, color=colors[3])]
CairoMakie.xlims!(ax_Ri, -1, 2)
CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NDE Prediction", "OceanTurb.jl Modified Pacalowski-Philander"])
supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
trim!(fig.layout)

record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=30) do n
    @info "Animating mp4 frame $n/$(length(times))..."
    frame[] = n
end
