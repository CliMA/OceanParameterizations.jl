using CairoMakie
using JLD2

NDE_DIR = "18sim_old"

filename = "train_wind_-5e-4_heating_-3e-8_new"

file = jldopen(joinpath("..\\final_results", NDE_DIR, filename, "profiles_fluxes_oceananigans.jld2"))
data = file["NDE_profile"]
close(file)

f_kpp = jldopen("../Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2")
kpp_profiles = f_kpp[filename[7:end]]
close(f_kpp)

frame = 1009


## TODO: Run Kpp fluxes
u_data = [
    data["truth_u"][:,frame],
    data["test_u_modified_pacanowski_philander"][:,frame],
    kpp_profiles["u"][:, frame],
    data["test_u"][:,frame],
]

v_data = [
    data["truth_v"][:,frame],
    data["test_v_modified_pacanowski_philander"][:,frame],
    kpp_profiles["v"][:, frame],
    data["test_v"][:,frame],
]

T_data = [
    data["truth_T"][:,frame],
    data["test_T_modified_pacanowski_philander"][:,frame],
    kpp_profiles["T"][:, frame],
    data["test_T"][:,frame],
]

uw_data = [
    data["truth_uw"][:,frame],
    data["test_uw_modified_pacanowski_philander"][:,frame],
    data["test_uw_kpp"][:,frame],
    # remember to change!!!
    data["test_uw"][:,frame],
]

vw_data = [
    data["truth_vw"][:,frame],
    data["test_vw_modified_pacanowski_philander"][:,frame],
    data["test_vw_kpp"][:,frame],
    # remember to change!!!
    data["test_vw"][:,frame],
]

wT_data = [
    data["truth_wT"][:,frame],
    data["test_wT_modified_pacanowski_philander"][:,frame],
    data["test_wT_kpp"][:,frame],
    # remember to change!!!
    data["test_wT"][:,frame],
]

uw_data .*= 1f4
vw_data .*= 1f4
wT_data .*= 1f5

Ri_data = [
    clamp.(data["truth_Ri"][:,frame], -1, 2),
    clamp.(data["test_Ri_modified_pacanowski_philander"][:,frame], -1, 2),
    clamp.(data["test_Ri_kpp"][:,frame], -1, 2),
    clamp.(data["test_Ri"][:,frame], -1, 2),
]

# losses_point_frames = [@lift [data[$frame]] for data in losses_data]

@inline function find_lims(datasets)
    return maximum(maximum.(datasets)), minimum(minimum.(datasets))
end

u_max, u_min = find_lims(u_data)
v_max, v_min = find_lims(v_data)
T_max, T_min = find_lims(T_data)

uw_max, uw_min = find_lims(uw_data)
vw_max, vw_min = find_lims(vw_data)
wT_max, wT_min = find_lims(wT_data)

##
fig = Figure(resolution=(2500, 1600), fontsize=50, figure_padding=30)

colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

temp_color = colors[2]
colors[2] = colors[4]
colors[4] = temp_color

zc = data["depth_profile"]
zf = data["depth_flux"]

ax_u = fig[1, 1] = Axis(fig, xlabel=L"$\overline{u}$ (m s$^{-1}$)", ylabel=L"z (m) $ $")
ax_v = fig[1, 2] = Axis(fig, xlabel=L"$\overline{v}$ (m s$^{-1}$)", ylabel=L"z (m) $ $")
ax_T = fig[1, 3] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_Ri = fig[1, 4] = Axis(fig, xlabel=L"Ri$ $", ylabel=L"z (m) $ $")
ax_uw = fig[2, 1] = Axis(fig, xlabel=L"$\overline{u\prime w\prime}$ (m$^2$ s$^{-2}$)", ylabel=L"z (m) $ $")
ax_vw = fig[2, 2] = Axis(fig, xlabel=L"$\overline{v\prime w\prime}$ (m$^2$ s$^{-2}$)", ylabel=L"z (m) $ $")
ax_wT = fig[2, 3] = Axis(fig, xlabel=L"$\overline{w\prime T\prime}$ ($\degree$C m s$^{-1}$)", ylabel=L"z (m) $ $")

axs = [ax_u, ax_v, ax_T, ax_Ri, ax_uw, ax_vw, ax_wT]

alpha=0.4
truth_linewidth = 12
linewidth = 7

# CairoMakie.xlims!(ax_u, u_min, u_max)
# CairoMakie.xlims!(ax_v, v_min, v_max)
# CairoMakie.xlims!(ax_T, T_min, T_max)
# CairoMakie.xlims!(ax_uw, uw_min, uw_max)
# CairoMakie.xlims!(ax_vw, vw_min, vw_max)
# CairoMakie.xlims!(ax_wT, wT_min, wT_max)
CairoMakie.xlims!(ax_Ri, -1, 2)

CairoMakie.ylims!(ax_u, minimum(zc), 0)
CairoMakie.ylims!(ax_v, minimum(zc), 0)
CairoMakie.ylims!(ax_T, minimum(zc), 0)
CairoMakie.ylims!(ax_uw, minimum(zf), 0)
CairoMakie.ylims!(ax_vw, minimum(zf), 0)
CairoMakie.ylims!(ax_wT, minimum(zf), 0)
CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

linkyaxes!(axs...)

hideydecorations!(ax_v, grid = false)
hideydecorations!(ax_T, grid = false)
hideydecorations!(ax_Ri, grid = false)
hideydecorations!(ax_vw, grid = false)
hideydecorations!(ax_wT, grid = false)

label_a = fig[1, 1, TopLeft()] = Label(fig, "A", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_b = fig[1, 2, TopLeft()] = Label(fig, "B", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_c = fig[1, 3, TopLeft()] = Label(fig, "C", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_d = fig[1, 4, TopLeft()] = Label(fig, "D", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_e = fig[2, 1, TopLeft()] = Label(fig, "E", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_f = fig[2, 2, TopLeft()] = Label(fig, "F", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
label_g = fig[2, 3, TopLeft()] = Label(fig, "G", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)

u_lines = [
        lines!(ax_u, u_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_u, u_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
]

v_lines = [
        lines!(ax_v, v_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
    [lines!(ax_v, v_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
]

T_lines = [
        lines!(ax_T, data["truth_T"][:,1], zc, linewidth=linewidth, color=colors[end], linestyle=:dash)
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

Legend(fig[3,1:4], T_lines, ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], 
       "Data Type", tellwidth=false, orientation=:horizontal, nbanks=2)
display(fig)
##
save("plots/profiles_fluxes.pdf", fig, px_per_unit=4, pt_per_unit=4)