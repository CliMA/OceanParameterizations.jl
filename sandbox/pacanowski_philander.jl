using CairoMakie

digit2superscript = Dict('1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹', '0' => '⁰', '-' => '⁻', '+' => '⁺')

superscript(n) = join(haskey(digit2superscript, d) ? digit2superscript[d] : d for d in string(n))

function plot_pacanowski_philander(; ν₀=1e-4, ν₁=1e-2, κ₀=1e-5, κ₁=1e-2, c=5, n=2)
    Ri = range(-0.199, 5, length=1001)
    Kᵁ = @. ν₀ + ν₁ / (1 + c*Ri)^n
    Kᵀ = @. κ₀ + κ₁ / (1 + c*Ri)^(n+1)

    fig = Figure(resolution = (1200, 800))
    ytickformat(ds) = ["10$(superscript(isinteger(d) ? Int(d) : d))" for d in ds]
    ax = fig[1, 1] = Axis(fig, title="Pacanowski & Philander", xlabel="Richardson number", ylabel="Diffusivity", xticks=0:5, ytickformat=ytickformat, yticks=-5:2:5)
    vline1 = vlines!(ax, [0.25], linewidth=3, color="lightgray")
    line1 = lines!(ax, Ri, log10.(Kᵁ), linewidth=3, label="Kᵁ", color="dodgerblue2")
    line2 = lines!(ax, Ri, log10.(Kᵀ), linewidth=3, label="Kᵀ", color="crimson")
    legend = fig[1, end+1] = Legend(fig, [line1, line2, vline1], ["Kᵁ", "Kᵀ", "Ri=¼"])
    xlims!(ax, extrema(Ri))
    ylims!(ax, -5, 5)
    save("pacanowski_philander.png", fig)

    return
end

