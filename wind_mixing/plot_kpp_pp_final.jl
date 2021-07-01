using Printf
using JLD2
using DataDeps
using OceanTurb
using CairoMakie
using WindMixing
using OceanParameterizations
using Images, FileIO
using ImageTransformations

include("modified_pacalowski_philander_model.jl")

train_files = ["wind_-5.5e-4_new"]
𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

NDE_profile_path = joinpath(pwd(), "final_results", "18sim_old", "test_$(train_files[1])", "profiles_fluxes_oceananigans.jld2")
file = jldopen(NDE_profile_path)
data = file["NDE_profile"]
close(file)

Nz = 32
Lz = 256
u₀ = 𝒟.u.coarse[:,1]
v₀ = 𝒟.v.coarse[:,1]
T₀ = 𝒟.T.coarse[:,1]
Fu = 𝒟.uw.coarse[end,1]
Fθ = 𝒟.wT.coarse[end,1]
f₀ = 1f-4
g = 9.80655f0
α = 2f-4
∂T₀∂z_bottom = (T₀[end-1] - T₀[end]) / (Lz / Nz)

frame = 1009

times = 𝒟.t

# constants_kpp = (f=f₀, α=α, g=g, Nz=Nz, H=Lz)
# BCs_kpp = (uw=(top=𝒟.uw.coarse[end, 1], bottom=𝒟.uw.coarse[1, 1]), 
#             vw=(top=𝒟.vw.coarse[end, 1], bottom=𝒟.uw.coarse[1, 1]), 
#             wT=(top=𝒟.wT.coarse[end, 1], bottom=𝒟.wT.coarse[1, 1]))
# ICs_kpp = (u=u₀, v=v₀, T=T₀)

# column_model_1D_kpp(constants_kpp, BCs_kpp, ICs_kpp, times, OceanTurb.KPP.Parameters())

constants_pp = OceanTurb.Constants(Float64, f=f₀)
parameters_pp = PacanowskiPhilander.Parameters()
model_pp = PacanowskiPhilander.Model(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants_pp, parameters=parameters_pp) 

model_pp.bcs[1].top = OceanTurb.FluxBoundaryCondition(Fu)
model_pp.bcs[3].top = OceanTurb.FluxBoundaryCondition(Fθ)

model_pp.solution[1].data[1:Nz] .= u₀
model_pp.solution[2].data[1:Nz] .= v₀
model_pp.solution[3].data[1:Nz] .= T₀

Δt = 1.0
Nt = length(times)

u_solution_pp = zeros(Nz, Nt)
v_solution_pp = zeros(Nz, Nt)
T_solution_pp = zeros(Nz, Nt)

uw_solution_pp = zeros(Nz+1, Nt)
vw_solution_pp = zeros(Nz+1, Nt)
wT_solution_pp = zeros(Nz+1, Nt)

Ri_solution_pp = zeros(Nz+1, Nt)

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
    OceanTurb.run_until!(model_pp, Δt, times[n])
    @info "Time = $(times[n])"

    u_solution_pp[:, n] .= model_pp.solution[1][1:Nz]
    v_solution_pp[:, n] .= model_pp.solution[2][1:Nz]
    T_solution_pp[:, n] .= model_pp.solution[3][1:Nz]

    uw_solution_pp[:, n] .= get_diffusive_flux(1, model_pp)[1:Nz+1]
    vw_solution_pp[:, n] .= get_diffusive_flux(2, model_pp)[1:Nz+1]
    wT_solution_pp[:, n] .= get_diffusive_flux(3, model_pp)[1:Nz+1]
    
    uw_solution_pp[Nz+1, n] = Fu
    wT_solution_pp[Nz+1, n] = Fθ

    # Ri_solution_pp[:, n] = get_richardson_number_profile(model_pp)[1:Nz+1]
end

D_face = Dᶠ(32, 256 / 32)

function local_richardson_profile(u, v, T, g, α)
    ∂u∂z = D_face * u
    ∂v∂z = D_face * v
    ∂T∂z = D_face * T
    Bz = g .* α .* ∂T∂z
    S² = ∂u∂z .^ 2 .+ ∂v∂z .^ 2
    return Bz ./ S²
end

for i in 1:size(Ri_solution_pp, 2)
    Ri_solution_pp[:,i] = local_richardson_profile(u_solution_pp[:,i], v_solution_pp[:,i], T_solution_pp[:,i], g, α)
end

##
# Get rid of ∞ and super large values.
Ri_solution_pp = clamp.(Ri_solution_pp, -1, 2)

u_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "u.png"))))
v_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "v.png"))))
T_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "T.png"))))
uw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "uw.png"))))
vw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "vw.png"))))
wT_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "wT.png"))))
Ri_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "Ri.png"))))
z_img = load(assetpath(joinpath(pwd(), "images_plot", "z.png")))

axis_images = (
    u = u_img,
    v = v_img,
    T = T_img,
    z = z_img,
    uw = uw_img,
    vw = vw_img,
    wT = wT_img,
    Ri = Ri_img,
)

FILE_PATH = joinpath(pwd(), "final_results", "base_kpp_pp")
fps=60

times_days = data["t"] ./ 86400

time_point = times_days[frame]

u_data = [
    data["truth_u"][:, frame],
    # data["test_u_modified_pacanowski_philander"],
    data["test_u_kpp"][:, frame],
    # data["test_u"],
    u_solution_pp[:, frame],
]

v_data = [
    data["truth_v"][:, frame],
    # data["test_v_modified_pacanowski_philander"],
    data["test_v_kpp"][:, frame],
    # data["test_v"],
    v_solution_pp[:, frame],
]

T_data = [
    data["truth_T"][:, frame],
    # data["test_T_modified_pacanowski_philander"],
    data["test_T_kpp"][:, frame],
    # data["test_T"],
    T_solution_pp[:, frame],
]

uw_data = [
    data["truth_uw"][:, frame],
    # data["test_uw_modified_pacanowski_philander"],
    data["test_uw_kpp"][:, frame],
    # data["test_uw"],
    uw_solution_pp[:, frame],
]

vw_data = [
    data["truth_vw"][:, frame],
    # data["test_vw_modified_pacanowski_philander"],
    data["test_vw_kpp"][:, frame],
    # data["test_vw"],
    vw_solution_pp[:, frame],
]

wT_data = [
    data["truth_wT"][:, frame],
    # data["test_wT_modified_pacanowski_philander"],
    data["test_wT_kpp"][:, frame],
    # data["test_wT"],
    wT_solution_pp[:, frame],
]

uw_data .*= 1f4
vw_data .*= 1f4
wT_data .*= 1f5

Ri_data = [
    clamp.(data["truth_Ri"][:, frame], -1, 2),
    # clamp.(data["test_Ri_modified_pacanowski_philander"], -1, 2),
    clamp.(data["test_Ri_kpp"][:, frame], -1, 2),
    # clamp.(data["test_Ri"], -1, 2),
    Ri_solution_pp[:, frame]
]

@inline function find_lims(datasets)
    return maximum(maximum.(datasets)), minimum(minimum.(datasets))
end

u_max, u_min = find_lims(u_data)
v_max, v_min = find_lims(v_data)
T_max, T_min = find_lims(T_data)

uw_max, uw_min = find_lims(uw_data)
vw_max, vw_min = find_lims(vw_data)
wT_max, wT_min = find_lims(wT_data)

# losses_max, losses_min = find_lims(losses_data)

train_parameters = data["train_parameters"]
ν₀ = train_parameters.ν₀
ν₋ = train_parameters.ν₋
ΔRi = train_parameters.ΔRi
Riᶜ = train_parameters.Riᶜ
Pr = train_parameters.Pr
loss_scalings = train_parameters.loss_scalings

# BC_str = @sprintf "Momentum Flux = %.1e m² s⁻², Temperature Flux = %.1e m s⁻¹ °C" data["truth_uw"][end, 1] maximum(data["truth_wT"][end, :])

# plot_title = @lift "Traditional Parameterisations: $BC_str, Time = $(round(times_days[$frame], digits=2)) days"

# fig = Figure(resolution=(1920, 1080))
fig = Figure(resolution=(1920, 960))

# colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

u_img = axis_images.u
v_img = axis_images.v
T_img = axis_images.T
uw_img = axis_images.uw
vw_img = axis_images.uw
wT_img = axis_images.uw
Ri_img = axis_images.uw
z_img = axis_images.z

zc = data["depth_profile"]
zf = data["depth_flux"]
zf_interior = zf[2:end-1]

rel_size = 30
# aspect = 1920 / 1080
aspect = 2


ax_u = fig[1, 2] = CairoMakie.Axis(fig)
ax_v = fig[1, 4] = CairoMakie.Axis(fig)

T_layout = fig[1:4, 5] = GridLayout()
colsize!(fig.layout, 5, CairoMakie.Relative(0.4))

ax_T = T_layout[1, 2] = CairoMakie.Axis(fig)
y_ax_T = T_layout[1,1] = CairoMakie.Axis(fig, aspect=DataAspect())
x_ax_T = T_layout[2,2] = CairoMakie.Axis(fig, aspect=DataAspect())

ax_Ri = fig[1, 7] = CairoMakie.Axis(fig)
ax_uw = fig[3, 2] = CairoMakie.Axis(fig)
ax_vw = fig[3, 4] = CairoMakie.Axis(fig)
ax_wT = fig[3, 7] = CairoMakie.Axis(fig)

y_ax_u = CairoMakie.Axis(fig[1,1], aspect=DataAspect())
y_ax_v = CairoMakie.Axis(fig[1,3], aspect=DataAspect())
y_ax_Ri = CairoMakie.Axis(fig[1,6], aspect=DataAspect())
y_ax_uw = CairoMakie.Axis(fig[3,1], aspect=DataAspect())
y_ax_vw = CairoMakie.Axis(fig[3,3], aspect=DataAspect())
y_ax_wT = CairoMakie.Axis(fig[3,6], aspect=DataAspect())

x_ax_u = CairoMakie.Axis(fig[2,2], aspect=DataAspect())
x_ax_v = CairoMakie.Axis(fig[2,4], aspect=DataAspect())
x_ax_Ri = CairoMakie.Axis(fig[2,7], aspect=DataAspect())
x_ax_uw = CairoMakie.Axis(fig[4,2], aspect=DataAspect())
x_ax_vw = CairoMakie.Axis(fig[4,4], aspect=DataAspect())
x_ax_wT = CairoMakie.Axis(fig[4,7], aspect=DataAspect())

hidedecorations!(y_ax_u)
hidedecorations!(y_ax_v)
hidedecorations!(y_ax_Ri)
hidedecorations!(y_ax_uw)
hidedecorations!(y_ax_vw)
hidedecorations!(y_ax_wT)

hidedecorations!(x_ax_u)
hidedecorations!(x_ax_v)
hidedecorations!(x_ax_Ri)
hidedecorations!(x_ax_uw)
hidedecorations!(x_ax_vw)
hidedecorations!(x_ax_wT)

hidespines!(y_ax_u)
hidespines!(y_ax_v)
hidespines!(y_ax_T)
hidespines!(y_ax_Ri)
hidespines!(y_ax_uw)
hidespines!(y_ax_vw)
hidespines!(y_ax_wT)

hidespines!(x_ax_u)
hidespines!(x_ax_v)
hidespines!(x_ax_T)
hidespines!(x_ax_Ri)
hidespines!(x_ax_uw)
hidespines!(x_ax_vw)
hidespines!(x_ax_wT)

image!(x_ax_u, axis_images.u)
image!(x_ax_v, axis_images.v)
image!(x_ax_Ri, axis_images.Ri)
image!(x_ax_uw, axis_images.uw)
image!(x_ax_vw, axis_images.vw)
image!(x_ax_wT, axis_images.wT)

image!(y_ax_u, axis_images.z)
image!(y_ax_v, axis_images.z)
image!(y_ax_Ri, axis_images.z)
image!(y_ax_uw, axis_images.z)
image!(y_ax_vw, axis_images.z)
image!(y_ax_wT, axis_images.z)

hidedecorations!(y_ax_T)
hidedecorations!(x_ax_T)
image!(x_ax_T, axis_images.T)
image!(y_ax_T, axis_images.z)

colsize!(T_layout, 1, CairoMakie.Relative(1 / rel_size * 1.5))
rowsize!(T_layout, 2, CairoMakie.Relative(1 / rel_size / aspect * 1.75))
colgap!(T_layout, Relative(1 / rel_size / aspect / 2))
rowgap!(T_layout, Relative(1 / rel_size / aspect))

rowsize!(fig.layout, 2, CairoMakie.Relative(1 / rel_size))
rowsize!(fig.layout, 4, CairoMakie.Relative(1 / rel_size))
colsize!(fig.layout, 1, CairoMakie.Relative(1 / rel_size / aspect))
colsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size / aspect))
colsize!(fig.layout, 6, CairoMakie.Relative(1 / rel_size / aspect))

colgap!(fig.layout, Relative(1 / rel_size / aspect / 2))
rowgap!(fig.layout, Relative(1 / rel_size / aspect))

alpha=0.5
truth_linewidth = 7
linewidth = 3

# CairoMakie.xlims!(ax_u, u_min, u_max)
# CairoMakie.xlims!(ax_v, v_min, v_max)
# CairoMakie.xlims!(ax_T, T_min, T_max)
# CairoMakie.xlims!(ax_uw, uw_min, uw_max)
# CairoMakie.xlims!(ax_vw, vw_min, vw_max)
# CairoMakie.xlims!(ax_wT, wT_min, wT_max)
CairoMakie.xlims!(ax_Ri, -1, 2)
# CairoMakie.xlims!(ax_losses, times[1], times[end])

CairoMakie.ylims!(ax_u, minimum(zc), 0)
CairoMakie.ylims!(ax_v, minimum(zc), 0)
CairoMakie.ylims!(ax_T, minimum(zc), 0)
CairoMakie.ylims!(ax_uw, minimum(zf), 0)
CairoMakie.ylims!(ax_vw, minimum(zf), 0)
CairoMakie.ylims!(ax_wT, minimum(zf), 0)
CairoMakie.ylims!(ax_Ri, minimum(zf), 0)
# CairoMakie.ylims!(ax_losses, losses_min, losses_max)

u_lines = [
     lines!(ax_u, u_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_u, u_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
]

v_lines = [
     lines!(ax_v, v_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_v, v_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
]

T_lines = [
    lines!(ax_T, data["truth_T"][:, 1], zc, linewidth=linewidth, color=colors[end], linestyle=:dot)
     lines!(ax_T, T_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_T, T_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(T_data)]
]

uw_lines = [
     lines!(ax_uw, uw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_uw, uw_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(uw_data)]
]

vw_lines = [
     lines!(ax_vw, vw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_vw, vw_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(vw_data)]
]

wT_lines = [
    lines!(ax_wT, wT_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
   [lines!(ax_wT, wT_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(wT_data)]
]

Ri_lines = [
     lines!(ax_Ri, Ri_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_Ri, Ri_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(Ri_data)]
]

axislegend(ax_T, T_lines, ["Initial Stratification", "Oceananigans.jl Large Eddy Simulation", "K-Profile Parameterisation", "Pacanowski-Philander Model"], "Data Type", position = :rb)


# supertitle = fig[0, :] = Label(fig, plot_title, textsize=25)

trim!(fig.layout)
fig

save("final_results/kpp_pp_comparison.png", fig, px_per_unit=4)