using CairoMakie
using Oceananigans
using FreeConvection

Nz = 32

ids_train = collect(1:9)
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

function color(id; alpha=1.0)
    colors = CairoMakie.Makie.wong_colors(alpha)
     1 <= id <= 9  && return colors[1]
    10 <= id <= 12 && return colors[2]
    13 <= id <= 15 && return colors[3]
    16 <= id <= 18 && return colors[4]
    19 <= id <= 21 && return colors[5]
    error("Invalid ID: $id")
end

function label(id)
     1 <= id <= 9  && return "training"
    10 <= id <= 12 && return "Qb interpolation"
    13 <= id <= 15 && return "Qb extrapolation"
    16 <= id <= 18 && return "N² interpolation"
    19 <= id <= 21 && return "N² extrapolation"
    error("Invalid ID: $id")
end

fig = Figure()

ax = fig[1, 1] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)", title="6 hours")

for id in FreeConvection.SIMULATION_IDS
    ds = datasets[id]
    T = ds["T"]
    zc = znodes(T)
    T_n = interior(T)[1, 1, :, 36] # 6 hours
    linestyle = 1 <= id <= 9 ? :solid : :dash
    lines!(ax, T_n, zc, color=color(id); linestyle)
end

ylims!(-256, 0)

ax = fig[2, 1] = Axis(fig, xlabel="Heat flux (m/s K)", ylabel="z (m)")

for id in FreeConvection.SIMULATION_IDS
    ds = datasets[id]
    wT = ds["wT"]
    # wT_param = ds["wT_param"]
    # wT_missing = ds["wT_missing"]
    zf = znodes(wT)

    # 6 hours
    wT_n = interior(wT)[1, 1, :, 36]
    # wT_param_n = interior(wT_param)[1, 1, :, 36]
    # wT_missing_n = interior(wT_missing)[1, 1, :, 36]

    linestyle = 1 <= id <= 9 ? :solid : :dash
    lines!(ax, wT_n, zf, color=color(id), linestyle=linestyle)
end

ylims!(-256, 0)

ax = fig[1, 2] = Axis(fig, xlabel="Temperature (°C)", title="4 days")

for id in FreeConvection.SIMULATION_IDS
    ds = datasets[id]
    T = ds["T"]
    zc = znodes(T)
    T_n = interior(T)[1, 1, :, 576] # 4 days
    linestyle = 1 <= id <= 9 ? :solid : :dash
    lines!(ax, T_n, zc, color=color(id); linestyle)
end

ylims!(-256, 0)

ax = fig[2, 2] = Axis(fig, xlabel="Heat flux (m/s K)", ylabel="z (m)")

for id in FreeConvection.SIMULATION_IDS
    ds = datasets[id]
    wT = ds["wT"]
    # wT_param = ds["wT_param"]
    # wT_missing = ds["wT_missing"]
    zf = znodes(wT)

    # 4 days
    wT_n = interior(wT)[1, 1, :, 576]
    # wT_param_n = interior(wT_param)[1, 1, :, 576]
    # wT_missing_n = interior(wT_missing)[1, 1, :, 576]

    linestyle = 1 <= id <= 9 ? :solid : :dash
    lines!(ax, wT_n, zf, color=color(id), linestyle=linestyle)
end

ylims!(-256, 0)

entry_ids = (1, 10, 13, 16, 19)
entries = [LineElement(color=color(id), linestyle = id == 1 ? :solid : :dash) for id in entry_ids]
labels = [label(id) for id in entry_ids]
Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

save("figure_training_data.png", fig, px_per_unit=2)
save("figure_training_data.pdf", fig, pt_per_unit=2)
