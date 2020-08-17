function plot_conservation(u, t; filename)
    N, Nt = size(u)
    Σu₀ = sum(u[:, 1])
    Σu = [sum(u[:, n]) for n in 1:Nt]

    p = plot(t, Σu .- Σu₀, linewidth=2, title="Conservation", label="")

    @info "Saving $filename..."
    savefig(p, filename)
end
