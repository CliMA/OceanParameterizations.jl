using CairoMakie
using FileIO
using JLD2
using Colors

f_kpp = jldopen("../Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2")

RESULTS_DIR = "..\\final_results"
NDE_DIR = "18sim_old"

function open_NDE_profile(filepath)
    file = jldopen(filepath)
    NDE_profile = file["NDE_profile"]
    close(file)
    return NDE_profile
end

files_training = [
    "train_wind_-5e-4_cooling_3e-8_new"
    "train_wind_-2e-4_cooling_3e-8_new"
    "train_wind_-5e-4_cooling_1e-8_new"

    "train_wind_-5e-4_heating_-3e-8_new"
    "train_wind_-2e-4_heating_-3e-8_new"
    "train_wind_-5e-4_heating_-1e-8_new"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

kpp_profiles_training = [
    f_kpp["$(train_file[7:end])/T"] for train_file in files_training
]

subplot_titles_training = [
    "Strong Wind, Strong Cooling"
    "Weak Wind, Strong Cooling"
    "Strong Wind, Weak Cooling"

    "Strong Wind, Strong Heating"
    "Weak Wind, Strong Heating"
    "Strong Wind, Weak Heating"
]

frame = 1009

T_datasets = [
    [
        data["truth_T"][:,frame],
        data["test_T_modified_pacanowski_philander"][:,frame],
        kpp_profiles_training[i][:, frame],
        data["test_T"][:,frame],
    ] for (i, data) in pairs(NDE_profiles_training)
]

zc = NDE_profiles_training[1]["depth_profile"]

@inline function find_lims(datasets)
    return maximum(maximum([maximum.(data) for data in datasets])), minimum(minimum([minimum.(data) for data in datasets]))
end

T_max, T_min = find_lims(T_datasets)

##
fig = Figure(resolution=(1280, 1000), fontsize=25)

colors = distinguishable_colors(length(T_datasets[1])+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

temp_color = colors[2]
colors[2] = colors[4]
colors[4] = temp_color

ax_T₁ = fig[1,1] = CairoMakie.Axis(fig, title=subplot_titles_training[1], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_T₂ = fig[1,2] = CairoMakie.Axis(fig, title=subplot_titles_training[2], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_T₃ = fig[1,3] = CairoMakie.Axis(fig, title=subplot_titles_training[3], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_T₄ = fig[2,1] = CairoMakie.Axis(fig, title=subplot_titles_training[4], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_T₅ = fig[2,2] = CairoMakie.Axis(fig, title=subplot_titles_training[5], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
ax_T₆ = fig[2,3] = CairoMakie.Axis(fig, title=subplot_titles_training[6], xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")

axs = [
    ax_T₁
    ax_T₂
    ax_T₃
    ax_T₄
    ax_T₅
    ax_T₆
]

linkyaxes!(axs...)
# linkxaxes!(axs...)

hideydecorations!(axs[2], grid = false)
hideydecorations!(axs[3], grid = false)
hideydecorations!(axs[5], grid = false)
hideydecorations!(axs[6], grid = false)

# rowsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size))
# rowsize!(fig.layout, 6, CairoMakie.Relative(1 / rel_size))

# colsize!(fig.layout, 1, CairoMakie.Relative(1 / rel_size / aspect))

# colsize!(fig.layout, 2, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
# colsize!(fig.layout, 3, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
# colsize!(fig.layout, 4, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))

# # # colsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size / aspect))
# # # colsize!(fig.layout, 5, CairoMakie.Relative(1 / rel_size / aspect))

# colgap!(fig.layout, Relative(1 / rel_size / aspect / 2))
# rowgap!(fig.layout, Relative(1 / rel_size / aspect / 3))

label_a = fig[1, 1, TopLeft()] = Label(fig, "A", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))
label_b = fig[1, 2, TopLeft()] = Label(fig, "B", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))
label_c = fig[1, 3, TopLeft()] = Label(fig, "C", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))
label_d = fig[2, 1, TopLeft()] = Label(fig, "D", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))
label_e = fig[2, 2, TopLeft()] = Label(fig, "E", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))
label_f = fig[2, 3, TopLeft()] = Label(fig, "F", fontsize = 20, font = :bold, halign = :right, padding = (0, 25, 10, 0))

alpha = 0.4
truth_linewidth = 10
linewidth = 4

@inline function make_lines(ax, data)
    lines = [
        lines!(ax, NDE_profiles_training[1]["truth_T"][:,1], zc, linestyle=:dash, color=colors[end], linewidth=linewidth);
        lines!(ax, data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax, data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(data)]
            ]
    return lines
end

T_lines_axs = [
    make_lines(axs[i], T_datasets[i]) for i in 1:length(T_datasets)
]

legend = fig[3, :] = CairoMakie.Legend(fig, T_lines_axs[1],
        ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"],
        orientation = :horizontal,
        nbanks = 2
)

# trim!(fig.layout)

display(fig)
##
save("plots/sample_T_profiles.pdf", fig, px_per_unit=4, pt_per_unit=4)