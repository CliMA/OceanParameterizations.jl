using CairoMakie

digit2superscript = Dict('1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹', '0' => '⁰', '-' => '⁻', '+' => '⁺')

superscript(n) = join(haskey(digit2superscript, d) ? digit2superscript[d] : d for d in string(n))

tanh_step(x) = (1 - tanh(x)) / 2

function plot_modified_pacanowski_philander(; ν₀=1e-4, ν₋=1, Riᶜ=0.25, ΔRi=0.1, Pr=1)
    Ri = range(-1, 3, length=1001)
    Kᵁ = @. ν₀ + ν₋ * tanh_step((Ri - Riᶜ) / ΔRi)
    Kᵀ = @. Kᵁ / Pr

    fig = Figure(resolution = (1200, 800))
    ytickformat(ds) = ["10$(superscript(isinteger(d) ? Int(d) : d))" for d in ds]
    ax = fig[1, 1] = Axis(fig, title="Modified Pacanowski & Philander", xlabel="Richardson number", ylabel="Diffusivity", xticks=-1:3, ytickformat=ytickformat, yticks=-5:1)
    vline1 = vlines!(ax, [0.25], linewidth=3, color="lightgray")
    line1 = lines!(ax, Ri, log10.(Kᵁ), linewidth=3, label="Kᵁ", color="dodgerblue2")
    line2 = lines!(ax, Ri, log10.(Kᵀ), linewidth=3, label="Kᵀ", color="crimson")
    legend = fig[1, end+1] = Legend(fig, [line1, line2, vline1], ["Kᵁ", "Kᵀ", "Ri=¼"])
    xlims!(ax, extrema(Ri))
    ylims!(ax, -5, 1)
    save("modified_pacanowski_philander.png", fig)

    return
end

