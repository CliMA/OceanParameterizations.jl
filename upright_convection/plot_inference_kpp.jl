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

function plot_pdfs(CSL, CNL, Cb_T, CKE; bins, filename="KPP_parameter_marginal_posteriors.png")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    axes[1, 1].hist(CSL, bins=bins, density=true)
    axes[1, 1].set_xlabel("CSL")
    axes[1, 1].set_xlim([0, 1])

    axes[1, 2].hist(CNL, bins=bins, density=true)
    axes[1, 2].set_xlabel("CNL")
    axes[1, 2].set_xlim([0, 8])

    axes[2, 1].hist(Cb_T, bins=bins, density=true)
    axes[2, 1].set_xlabel("Cb_T")
    axes[2, 1].set_xlim([0, 6])

    axes[2, 2].hist(CKE, bins=bins, density=true)
    axes[2, 2].set_xlabel("CKE")
    axes[2, 2].set_xlim([0, 5])

    @info "Saving $filename..."
    plt.savefig(filename)

    return nothing
end

function plot_kpp_uncertainty(T, zC, solutions, zC_cs; filename="KPP_uncertainty.png")
    solutions = cat(solutions..., dims=3)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)

    time_indices = (1, 36, 144, 432, 1152)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")

    for (n, color) in zip(time_indices, colors)
        ax.plot(T[n, :], zC, color=color)

        T_mean = mean(solutions[:, n, :], dims=2)[:]
        T_std  = std(solutions[:, n, :], dims=2)[:]

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

    ax.legend(custom_lines, ["LES", "KPP mean", "KPP uncertainty"], loc="lower right", frameon=false)

    @info "Saving $filename..."
    plt.savefig(filename)

    return nothing
end

function make_KPP_inference_plots(bson_filepath)
    data = BSON.load(bson_filepath)

    CSL, CNL, Cb_T, CKE = data[:CSL], data[:CNL], data[:Cb_T], data[:CKE]
    T, zC, solutions, zC_cs = data[:T], data[:zC], data[:solutions], data[:zC_cs]

    plot_pdfs(CSL, CNL, Cb_T, CKE, bins=20)
    plot_kpp_uncertainty(T, zC, solutions, zC_cs)
end

make_KPP_inference_plots("inferred_KPP_parameters.bson")
