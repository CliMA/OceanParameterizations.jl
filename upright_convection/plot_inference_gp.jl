using Statistics
using OceanTurb
using JLD2
using BSON
using OceanParameterizations

# Headless plotting with PyPlot
ENV["MPLBACKEND"] = "Agg"

import PyPlot
const plt = PyPlot
const Line2D = plt.matplotlib.lines.Line2D
const Patch = plt.matplotlib.patches.Patch

function plot_hyperparameter_pdfs(ls, σ²s; bins, filename="GP_hyperparameter_posteriors.png")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    fig.suptitle("Squared exponential kernel")

    axes[1, 1].hist(ls, bins=bins, density=true)
    axes[1, 1].set_xlabel("lengthscale l")

    axes[2, 1].hist(σ²s, bins=bins, density=true)
    axes[2, 1].set_xlabel("variance σ²")

    @info "Saving $filename..."
    plt.savefig(filename)

    return nothing
end

function plot_gp_uncertainty(T, zC, solutions, zC_cs; filename="GP_uncertainty.png")
    Ns, Nt, Nz = length(solutions), length(solutions[1]), length(solutions[1][1])
    sols = zeros(Ns, Nt, Nz)
    for s in 1:Ns, n in 1:Nt, k in 1:Nz
        sols[s, n, k] = solutions[s][n][k]
    end

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)

    time_indices = (1, 36, 144, 432, 1152)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")

    for (n, color) in zip(time_indices, colors)
        ax.plot(T[n, :], zC, color=color)

        T_mean = mean(sols[:, n, :], dims=1)[:]
        T_std  = std(sols[:, n, :], dims=1)[:]

        ax.plot(T_mean, zC_cs, color=color, linestyle="--")
        ax.fill_betweenx(zC_cs, T_mean - T_std, T_mean + T_std, color=color, alpha=0.5)
    end

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("z (m)")
    ax.set_xlim([19, 20])
    ax.set_ylim([-100, 0])

    ax.text(19.98, 1, "0 hours", color="tab:blue",   rotation=45)
    ax.text(19.88, 1, "6 hours", color="tab:orange", rotation=45)
    ax.text(19.77, 1, "1 day",   color="tab:green",  rotation=45)
    ax.text(19.62, 1, "4 days",  color="tab:red",    rotation=45)
    ax.text(19.38, 1, "8 days",  color="tab:purple", rotation=45)

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="-"),
        Line2D([0], [0], color="black", linestyle="--"),
        Patch(facecolor="black", alpha=0.5)
    ]

    ax.legend(custom_lines, ["LES", "GP mean", "GP uncertainty"], loc="lower right", frameon=false)

    @info "Saving $filename..."
    plt.savefig(filename)

    return nothing
end

data = BSON.load("inferred_GP_hyperparameters.bson")

ls, σ²s, solutions = data[:l], data[:σ²], data[:solutions]
T, zC, zC_cs = data[:T], data[:zC], data[:zC_cs]
plot_hyperparameter_pdfs(ls, σ²s, bins=20)
plot_gp_uncertainty(T, zC, solutions, zC_cs)
