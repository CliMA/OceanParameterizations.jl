using Plots
using LaTeXStrings

tanh_step(x) = (1 - tanh(x)) / 2

ν₀ = 1f-4
ν₋ = 0.1f0
ν₂ = 1f0

Riᶜ = 0.25
ΔRi₊ = 0.1
ΔRi₋ = 0.1

Ris = -1:0.001:2
mpp(Ri, ν₀, ν₋, Riᶜ, ΔRi) = ν₀ + ν₋ * tanh_step((Ri - Riᶜ) / ΔRi)
mpp_ca(Ri, ν₀, ν₋, ν₂, Riᶜ, ΔRi₊) = ν₀ + ν₋ * tanh_step((Ri - Riᶜ) / ΔRi₊) + ν₂ * tanh_step((Ri + Riᶜ) / ΔRi₋)

mpp_str = "Modified Pac-Phil"
mpp_ca_str = "New Modified Pac-Phil"

plot(Ris, mpp.(Ris, ν₀, ν₋, Riᶜ, ΔRi₊), label=mpp_str)
plot!(Ris, mpp_ca.(Ris, ν₀, ν₋, ν₂, Riᶜ, ΔRi₊), label=mpp_ca_str)
title!("Diffusivity Profiles Comparison")
savefig("Output/diffusivity_schemes_comparison.pdf")
# color_palette = distinguishable_colors(2, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

# fig = Figure()
# ax = Axis(fig[1, 1], yscale=log10)
# lines!(ax, Ris, mpp.(Ris, ν₀, ν₋, Riᶜ, ΔRi), color=color_palette[1], label=L"\nu_0")
# lines!(ax, Ris, mpp_ca.(Ris, ν₀, ν₋, ν₂, Riᶜ, ΔRi), color=color_palette[2])
# axislegend(ax)
# current_figure()
