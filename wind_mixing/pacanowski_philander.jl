using Printf
using JLD2
using DataDeps
using OceanTurb
using CairoMakie

ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/"

dd1 = DataDep("strong_wind",
              "proto-LESbrary.jl 2-day suite (strong wind)",
              joinpath(ENGAGING_LESBRARY_DIR,
                       "three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind",
                       "three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2"))

dd2 = DataDep("strong_wind_no_rotation",
              "proto-LESbrary.jl 2-day suite (strong wind, no rotation)",
              joinpath(ENGAGING_LESBRARY_DIR,
                       "three_layer_constant_fluxes_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation",
                       "three_layer_constant_fluxes_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation_statistics.jld2"))

DataDeps.register(dd1)
DataDeps.register(dd2)

# ds = jldopen(datadep"strong_wind/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2")
ds = jldopen(datadep"strong_wind_no_rotation/three_layer_constant_fluxes_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation_statistics.jld2")

Nz = ds["grid/Nz"]
Lz = ds["grid/Lz"]
u₀ = ds["timeseries/u/0"][:]
v₀ = ds["timeseries/v/0"][:]
T₀ = ds["timeseries/T/0"][:]
Fu = ds["boundary_conditions/u_top"]
Fθ = ds["boundary_conditions/θ_top"]

f₀ = ds["parameters/coriolis_parameter"]
constants = OceanTurb.Constants(Float64, f=f₀)
parameters = PacanowskiPhilander.Parameters()
model = PacanowskiPhilander.Model(N=Nz, L=Lz, stepper=:BackwardEuler, constants=constants, parameters=parameters)

model.bcs[1].top = OceanTurb.FluxBoundaryCondition(Fu)
model.bcs[3].top = OceanTurb.FluxBoundaryCondition(Fθ)

model.solution[1].data[1:Nz] .= u₀
model.solution[2].data[1:Nz] .= v₀
model.solution[3].data[1:Nz] .= T₀

Δt = 600.0
times = [ds["timeseries/t/$i"] for i in keys(ds["timeseries/t"])]
Nt = length(times)

U_solution = zeros(Nz, Nt)
V_solution = zeros(Nz, Nt)
T_solution = zeros(Nz, Nt)
U_flux = zeros(Nz+1, Nt)
V_flux = zeros(Nz+1, Nt)
T_flux = zeros(Nz+1, Nt)

function get_diffusive_flux(field_index, model)
    flux = FaceField(model.grid)
    field = model.solution[field_index]
    K = model.timestepper.eqn.K[field_index]
    for i in interiorindices(flux)
        @inbounds flux[i] = - K(model, i) * ∂z(field, i)
    end
    return flux
end

for n in 1:Nt
    OceanTurb.run_until!(model, Δt, times[n])
    @info "Time = $(times[n])"

    U_solution[:, n] .= model.solution[1][1:Nz]
    V_solution[:, n] .= model.solution[2][1:Nz]
    T_solution[:, n] .= model.solution[3][1:Nz]

    U_flux[:, n] .= get_diffusive_flux(1, model)[1:Nz+1]
    V_flux[:, n] .= get_diffusive_flux(2, model)[1:Nz+1]
    T_flux[:, n] .= get_diffusive_flux(3, model)[1:Nz+1]
    U_flux[Nz+1, n] = Fu
end

## Plot!

U_LES_solution = zeros(Nz, Nt)
V_LES_solution = zeros(Nz, Nt)
T_LES_solution = zeros(Nz, Nt)

U′W′_LES_solution = zeros(Nz+1, Nt)
V′W′_LES_solution = zeros(Nz+1, Nt)
W′T′_LES_solution = zeros(Nz+1, Nt)

for (n, iter) in zip(1:length(times), keys(ds["timeseries/t"]))
    U_LES_solution[:, n] = ds["timeseries/u/$iter"]
    V_LES_solution[:, n] = ds["timeseries/v/$iter"]
    T_LES_solution[:, n] = ds["timeseries/T/$iter"]
    
    U′W′_LES_solution[:, n] = ds["timeseries/wu/$iter"]
    V′W′_LES_solution[:, n] = ds["timeseries/wv/$iter"]
    W′T′_LES_solution[:, n] = ds["timeseries/wT/$iter"]
end

zc = model.grid.zc
zf = model.grid.zf

frame = Node(1)
plot_title = @lift @sprintf("Pacanowski-Philander wind-mixing: time = %s", prettytime(times[$frame]))

U = @lift U_solution[:, $frame]
V = @lift V_solution[:, $frame]
T = @lift T_solution[:, $frame]

U′W′ = @lift U_flux[:, $frame]
V′W′ = @lift V_flux[:, $frame]
W′T′ = @lift T_flux[:, $frame]

U_LES = @lift U_LES_solution[:, $frame]
V_LES = @lift V_LES_solution[:, $frame]
T_LES = @lift T_LES_solution[:, $frame]

U′W′_LES = @lift U′W′_LES_solution[:, $frame]
V′W′_LES = @lift V′W′_LES_solution[:, $frame]
W′T′_LES = @lift W′T′_LES_solution[:, $frame]

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

ax_UW = fig[2, 1] = Axis(fig, xlabel="U′W′ (m²/s²)", ylabel="z (m)")
l1_UW = lines!(ax_UW, U′W′, zf, linewidth=3, color="dodgerblue2")
l2_UW = lines!(ax_UW, U′W′_LES, zf, linewidth=3, color="crimson")
xlims!(ax_UW, extrema(U_flux))
ylims!(ax_UW, -Lz, 0)

ax_VW = fig[2, 2] = Axis(fig, xlabel="V′W′ (m²/s²)", ylabel="z (m)")
l1_VW = lines!(ax_VW, V′W′, zf, linewidth=3, color="dodgerblue2")
l2_VW = lines!(ax_VW, V′W′_LES, zf, linewidth=3, color="crimson")
xlims!(ax_VW, all(V_flux .== 0) ? (-1, 1) : extrema(V_flux))
ylims!(ax_VW, -Lz, 0)

ax_WT = fig[2, 3] = Axis(fig, xlabel="W′T′ (m/s ⋅ °C)", ylabel="z (m)")
l1_WT = lines!(ax_WT, W′T′, zf, linewidth=3, color="dodgerblue2")
l2_WT = lines!(ax_WT, W′T′_LES, zf, linewidth=3, color="crimson")
xlims!(ax_WT, extrema(W′T′_LES_solution))
ylims!(ax_WT, -Lz, 0)

legend = fig[1, 4] = Legend(fig, [l1_U, l2_U], ["Pacanowski-Philander", "Oceananigans.jl LES"])

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
trim!(fig.layout)

record(fig, "pacanowski_philander.mp4", 1:length(times), framerate=15) do n
    @info "Animating Pacanowski-Philander wind-mixing frame $n/$(length(times))..."
    frame[] = n
end
