CSL_hist = histogram(CSL,  bins=range(0, 1, length=10), xlabel="CSL",  label="")
CNL_hist = histogram(CNL,  bins=range(0, 8, length=10), xlabel="CNL",  label="")
CbT_hist = histogram(Cb_T, bins=range(0, 6, length=10), xlabel="Cb_T", label="")
CKE_hist = histogram(CKE,  bins=range(0, 5, length=10), xlabel="CKE",  label="")

p = plot(CSL_hist, CNL_hist, CbT_hist, CKE_hist, layout=(2,2), dpi=200)
savefig(p, "KPP_parameters_marginal_posteriors.png")

anim = @animate for n=1:5:Nt
    title = @sprintf("Little KPP ensemble: %.2f days", t[n] / 86400)

    KPP_solution, KPP_zC = traces[1].retval
    p = plot(KPP_solution[:, n], KPP_zC, linewidth=2,
         xlim=(19, 20), ylim=(-100, 0), label="",
         xlabel="Temperature (C)", ylabel="Depth (z)",
         title=title, dpi=200, show=false)

    for i in 2:10
        KPP_solution, KPP_zC = traces[i].retval
        plot!(p, KPP_solution[:, n], KPP_zC, linewidth=2, label="")
    end

    plot!(p, T[n, :], zC, linewidth=2, label="LES", legend=:bottomright)
end

gif(anim, "deepening_mixed_layer_KPP_ensemble.gif", fps=15)

function plot_pdfs(CSL, CNL, Cb_T, CKE; bins)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))

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

    plt.savefig("KPP_parameter_marginal_posteriors.png")

    return nothing
end

plot_pdfs(CSL, CNL, Cb_T, CKE, bins=20)
