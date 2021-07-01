using CairoMakie: Cairo
using CairoMakie
using JLD2

FILE_NAME = "NDE_18sim_windcooling_windheating_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8_1e-4"
DATA_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
OUTPUT_PATH = "C:\\Users\\xinle\\Documents\\OceanParameterizations.jl"

file = jldopen(DATA_PATH, "r")
train_parameters = file["training_info/parameters"]
close(file)

ν₀ = train_parameters["ν₀"]
ν₋ = train_parameters["ν₋"]
ΔRi = train_parameters["ΔRi"]
Riᶜ = train_parameters["Riᶜ"]
Pr = train_parameters["Pr"]

function diffusivity(Ri, ν₀, ν₋, ΔRi, Riᶜ, Pr)
    tanh_step(x) = (1 - tanh(x)) / 2
    ν = ν₀ + ν₋ * tanh_step((Ri - Riᶜ) / ΔRi)
    return ν
end

function diffusivity_T(Ri, ν₀, ν₋, ΔRi, Riᶜ, Pr)
    return diffusivity(Ri, ν₀, ν₋, ΔRi, Riᶜ, Pr) / Pr
end

Ris = range(-2, 2, length=1000)

ν = diffusivity.(Ris, ν₀, ν₋, ΔRi, Riᶜ, Pr)
ν_T = diffusivity_T.(Ris, ν₀, ν₋, ΔRi, Riᶜ, Pr)

fig = Figure(resolution=(1500, 750))
ax = fig[1,1] = CairoMakie.Axis(fig, yscale=log10, xlabel="Ri", ylabel="Diffusivity/ m² s⁻¹")


ν_line = CairoMakie.lines!(ax, Ris, ν)
ν_T_line = CairoMakie.lines!(ax, Ris, ν_T)

axislegend(ax, [ν_line, ν_T_line], ["Momentum Diffusivity", "Temperature Diffusivity"])
fig

save(joinpath("C:\\Users\\xinle\\Documents\\OceanParameterizations.jl", "$(FILE_NAME)_mpp.png"), fig, px_per_unit=4)