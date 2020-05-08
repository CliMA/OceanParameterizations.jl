# Headless plotting with PyPlot
ENV["MPLBACKEND"] = "Agg"

import PyPlot
const plt = PyPlot

function plot_LES_figure(T, zC, t; filename="LES_figure.png")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.75), dpi=200)

    ax.plot(T[1, :], zC, label="t = 0")
    ax.plot(T[36, :], zC, label=@sprintf("t = %d hours", t[36]/3600))
    ax.plot(T[144, :], zC, label=@sprintf("t = %d day", t[144]/86400))
    ax.plot(T[432, :], zC, label=@sprintf("t = %d days", t[432]/86400))
    ax.plot(T[1152, :], zC, label=@sprintf("t = %d days", t[1152]/86400))

    ax.legend(loc="lower right", frameon=false)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("z (m)")
    ax.set_xlim([19, 20])
    ax.set_ylim([-100, 0])

    @info "Saving $filename..."
    plt.savefig(filename, dpi="figure", bbox_inches="tight")

    return nothing
end

function animate_LES_solution(T, zC, t; filename="deepening_mixed_layer.gif", skip=5)
    Nt, N = size(T)

    anim = @animate for n in 1:skip:Nt
        title = @sprintf("Deepening mixed layer: %.2f days", t[n] / 86400)
        plot(T[n, :], zC, linewidth=2,
             xlim=(19, 20), ylim=(-100, 0), label="",
             xlabel="Temperature (C)", ylabel="Depth (z)",
             title=title, show=false)
    end

    @info "Saving $filename..."
    gif(anim, "deepening_mixed_layer.gif", fps=15)

    return nothing
end
