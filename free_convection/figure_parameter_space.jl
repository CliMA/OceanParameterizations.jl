using CairoMakie

Qbs = 1e-8 .* [
    1, 3,   5,
    1, 3,   5,
    1, 3,   5,
    4, 2,   4,
    6, 0.5, 6,
    1, 3,   5,
    1, 3,   5
]

N²s = 1e-5 .* [
    1,    1,    1,
    1.5,  1.5,  1.5,
    2,    2,    2,
    1,    1.5,  2,
    1,    1.5,  2,
    1.25, 1.25, 1.75,
    0.5,  0.5,  3
]

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
ax = fig[1, 1] = Axis(fig, xlabel="Surface buoyancy flux (m²/s³) ", ylabel="Thermocline stratification (1/s²)")

for (n, (Qb, N²)) in enumerate(zip(Qbs, N²s))
    scatter!(ax, [Qb], [N²], color=color(n))
    text!(ax, " " * string(n), position=(Qb, N²), color=color(n), align=(:left, :center))
end

band!(ax, [1e-8, 5e-8], 1e-5, 2e-5, color=color(1, alpha=0.25))

entry_ids = (1, 10, 13, 16, 19)
entries = [MarkerElement(marker=:circle, color=color(id), strokecolor=:transparent, markersize=15) for id in entry_ids]
labels = [label(id) for id in entry_ids]
Legend(fig[1, 2], entries, labels, framevisible=false)

xlims!(0.25e-8, 7e-8)

save("parameter_space.png", fig, px_per_unit=2)
save("parameter_space.pdf", fig, pt_per_unit=2)
