using Statistics
using OceanTurb
using JLD2
using BSON

# Headless plotting with PyPlot
# ENV["MPLBACKEND"] = "Agg"

import PyPlot
const plt = PyPlot
const Line2D = plt.matplotlib.lines.Line2D
const Patch = plt.matplotlib.patches.Patch

data = BSON.load("inferred_KPP_parameters.bson")
CSL, CNL, Cb_T, CKE = data[:CSL], data[:CNL], data[:Cb_T], data[:CKE]

function plot_pdfs(CSL, CNL, Cb_T, CKE; bins)
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

    plt.savefig("KPP_parameter_marginal_posteriors.png")

    return nothing
end

plot_pdfs(CSL, CNL, Cb_T, CKE, bins=20)

"""
avg(Φ, n)
# Description
- Average a field down by n.
- Requires field to have evenly spaced points. Size of N leq length(Φ).
- Furthermore requires
# Arguments
- `Φ` :(vector) The field, an array
- `n` :(Int) number of grid points to average down to.
# Return
- `Φ2` :(vector) The field with values averaged, an array
"""
function avg(Φ, n)
    m = length(Φ)
    scale = Int(floor(m/n))
    if ( abs(Int(floor(m/n)) - m/n) > eps(1.0))
        return error
    end
    Φ2 = zeros(n)
    for i in 1:n
        Φ2[i] = 0
            for j in 1:scale
                Φ2[i] += Φ[scale*(i-1) + j] / scale
            end
    end
    return Φ2
end

function free_convection_model(parameters, constants, N, L, Δt, times, T₀, FT, ∂T∂z)
    model = KPP.Model(N=N, H=L, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    # Coarse grain initial condition from LES and set equal
    # to initial condition of parameterization.
    model.solution.T.data[1:N] .= avg(T₀, N)

    # Set boundary conditions
    model.bcs.T.top = FluxBoundaryCondition(FT)
    model.bcs.T.bottom = GradientBoundaryCondition(∂T∂z)

    Nt = length(times)
    solution = zeros(N, Nt)

    # loop the model
    for n in 1:Nt
        run_until!(model, Δt, times[n])
        @. solution[:, n] = model.solution.T[1:N]
    end

    return solution, model.grid.zc
end

file = jldopen("free_convection_profiles.jld2")

Is = keys(file["timeseries/t"])

zC = file["grid/zC"]
Nz = file["grid/Nz"]
Lz = file["grid/Lz"]
Nt = length(Is)

t = zeros(Nt)
T = T_data = zeros(Nt, Nz)
wT = zeros(Nt, Nz)

for (i, I) in enumerate(Is)
    t[i] = file["timeseries/t/$I"]
    T[i, :] = file["timeseries/T/$I"][1, 1, 2:Nz+1]
end

# Physical constants
ρ₀ = file["parameters/density"]
cₚ = file["parameters/specific_heat_capacity"]
f  = file["parameters/coriolis_parameter"]
α  = file["buoyancy/equation_of_state/α"]
β  = file["buoyancy/equation_of_state/β"]
g  = file["buoyancy/gravitational_acceleration"]

constants = Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)

# OceanTurb parameters
N = 16
L = file["grid/Lz"]
Δt = 60

Q = parse(Float64, file["parameters/surface_cooling"])
FT = -Q / (ρ₀*cₚ)
∂T∂z = file["parameters/temperature_gradient"]
T₀ = T[1, :]

sols = []
samples = length(CSL)
for i in 1:samples
    global z
    @info "$i"
    parameters = KPP.Parameters(CSL=CSL[i], CNL=CNL[i], Cb_T=Cb_T[i], CKE=CKE[i])
    sol, z = free_convection_model(parameters, constants, N, L, Δt, t, T₀, FT, ∂T∂z)
    push!(sols, sol)
end
sols = cat(sols..., dims=3)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)

for (n, color) in zip((1, 36, 144, 432, 1152), ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"))
    T_mean = mean(sols[:, n, :], dims=2)[:]
    T_std  = std(sols[:, n, :], dims=2)[:]

    ax.plot(T[n, :], zC, color=color)
    ax.plot(T_mean, z, color=color, linestyle="--")
    ax.fill_betweenx(z, T_mean - T_std, T_mean + T_std, color=color, alpha=0.5)
end

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("z (m)")
ax.set_xlim([19, 20])
ax.set_ylim([-100, 0])

ax.text(19.98, 1, "0 hours", color="tab:blue", rotation=45)
ax.text(19.88, 1, "6 hours", color="tab:orange", rotation=45)
ax.text(19.77, 1, "1 day", color="tab:green", rotation=45)
ax.text(19.62, 1, "4 days", color="tab:red", rotation=45)
ax.text(19.38, 1, "8 days", color="tab:purple", rotation=45)

custom_lines = [
    Line2D([0], [0], color="black", linestyle="-"),
    Line2D([0], [0], color="black", linestyle="--"),
    Patch(facecolor="black", alpha=0.5)
]

ax.legend(custom_lines, ["LES", "KPP mean", "KPP uncertainty"], loc="lower right", frameon=false)

plt.savefig("KPP_uncertainty.png")
