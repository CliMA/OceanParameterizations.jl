using Printf
using JLD2
using DataDeps
using OceanTurb
using CairoMakie

using Oceananigans.Utils

ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/"

dd = DataDep("strong_wind",
             "proto-LESbrary.jl 2-day suite (strong wind)",
             joinpath(ENGAGING_LESBRARY_DIR,
                      "three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind",
                      "three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2"))

DataDeps.register(dd)

ds = jldopen(datadep"strong_wind/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2")

Nz = ds["grid/Nz"]
Lz = ds["grid/Lz"]
u₀ = ds["timeseries/u/0"][:]
v₀ = ds["timeseries/v/0"][:]
T₀ = ds["timeseries/T/0"][:]
Fu = ds["boundary_conditions/u_top"]

f₀ = ds["parameters/coriolis_parameter"]
constants = OceanTurb.Constants(Float64, f=f₀)
parameters = PacanowskiPhilander.Parameters()
model = PacanowskiPhilander.Model(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=parameters)

model.bcs[1].top = OceanTurb.FluxBoundaryCondition(Fu)

model.solution[1].data[1:Nz] .= u₀
model.solution[2].data[1:Nz] .= v₀
model.solution[3].data[1:Nz] .= T₀

Δt = 600.0
times = [ds["timeseries/t/$i"] for i in keys(ds["timeseries/t"])]
Nt = length(times)

u_solution = zeros(Nz, Nt)
v_solution = zeros(Nz, Nt)
T_solution = zeros(Nz, Nt)
u_flux = zeros(Nz+1, Nt)
v_flux = zeros(Nz+1, Nt)
T_flux = zeros(Nz+1, Nt)

for n in 1:Nt
    @info "Time = $(times[n])"
    OceanTurb.run_until!(model, Δt, times[n])

    u_solution[:, n] .= model.solution[1][1:Nz]
    v_solution[:, n] .= model.solution[2][1:Nz]
    T_solution[:, n] .= model.solution[3][1:Nz]

    # u_flux[:, n] .= OceanTurb.diffusive_flux(:u, model)[1:N+1]
    # v_flux[:, n] .= OceanTurb.diffusive_flux(:v, model)[1:N+1]
    # T_flux[:, n] .= OceanTurb.diffusive_flux(:T, model)[1:N+1]
    # flux[N+1, n] = FT
end

## Plot!

U_LES_solution = zeros(Nz, Nt)
V_LES_solution = zeros(Nz, Nt)
T_LES_solution = zeros(Nz, Nt)

for (n, iter) in zip(1:length(times), keys(ds["timeseries/t"]))
    U_LES_solution[:, n] = ds["timeseries/u/$iter"]
    V_LES_solution[:, n] = ds["timeseries/v/$iter"]
    T_LES_solution[:, n] = ds["timeseries/T/$iter"]
end

zc = model.grid.zc

frame = Node(1)
plot_title = @lift @sprintf("Pacanowski-Philander wind-mixing: time = %s", prettytime(times[$frame]))

U = @lift u_solution[:, $frame]
V = @lift v_solution[:, $frame]
T = @lift T_solution[:, $frame]

U_LES = @lift U_LES_solution[:, $frame]
V_LES = @lift V_LES_solution[:, $frame]
T_LES = @lift T_LES_solution[:, $frame]

fig = Figure(resolution=(1920, 1080))

ax_U = fig[1, 1] = Axis(fig, xlabel="U (m/s)", ylabel="z (m)")
l1_U = lines!(ax_U, U, zc, linewidth=3, color="dodgerblue2")
l2_U = lines!(ax_U, U_LES, zc, linewidth=3, color="crimson")
xlims!(ax_U, -1, 1)
ylims!(ax_U, -Lz, 0)

ax_V = fig[1, 2] = Axis(fig, xlabel="V (m/s)", ylabel="z (m)")
l1_V = lines!(ax_V, V, zc, linewidth=3, color="dodgerblue2")
l2_V = lines!(ax_V, V_LES, zc, linewidth=3, color="crimson")
xlims!(ax_V, -1, 1)
ylims!(ax_V, -Lz, 0)

ax_T = fig[1, 3] = Axis(fig, xlabel="T (°C)", ylabel="z (m)")
l1_T = lines!(ax_T, T, zc, linewidth=3, color="dodgerblue2")
l2_T = lines!(ax_T, T_LES, zc, linewidth=3, color="crimson")
ylims!(ax_T, -Lz, 0)

legend = fig[1, 4] = Legend(fig, [l1_U, l2_U], ["Pacanowski-Philander", "Oceananigans.jl LES"])

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
trim!(fig.layout)

record(fig, "pacanowski_philander.mp4", 1:length(times), framerate=15) do n
    @info "Animating Pacanowski-Philander wind-mixing frame $n/$(length(times))..."
    frame[] = n
end
