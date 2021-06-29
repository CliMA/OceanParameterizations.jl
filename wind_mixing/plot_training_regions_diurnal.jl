using CairoMakie
using Colors
using Makie.GeometryBasics
using JLD2

training_data = [
    (-5e-4, 5e-8),
    (-3.5e-4, 5e-8),
    (-2e-4, 5e-8),
]

train_files = [
    "wind_-5e-4_diurnal_5e-8"  
    "wind_-3.5e-4_diurnal_5e-8"
    "wind_-2e-4_diurnal_5e-8"  
]

interpolating_data = [
    (-4.5e-4, 4e-8),
    (-4.5e-4, 2e-8),
    (-3e-4, 4e-8),
    (-3e-4, 2e-8),
]

interpolating_files = [
    "wind_-4.5e-4_diurnal_4e-8"
    "wind_-4.5e-4_diurnal_2e-8"
    "wind_-3e-4_diurnal_4e-8"  
    "wind_-3e-4_diurnal_2e-8"  
]

extrapolating_data_diurnal = [
    (-5.5e-4, 5.5e-8),
    (-1.5e-4, 5.5e-8),
]

extrapolating_diurnal_files = [
    "wind_-5.5e-4_diurnal_5.5e-8"
    "wind_-1.5e-4_diurnal_5.5e-8"
]

extrapolating_data = [
    (-1.5e-4, 3.5e-8),
    (-5.5e-4, 3.5e-8),
    (-5.5e-4, 0),
    (-5.5e-4, -3.5e-8),
    (-1.5e-4, -3.5e-8),
]

extrapolating_files = [
    "wind_-1.5e-4_cooling_3.5e-8" 
    "wind_-5.5e-4_cooling_3.5e-8" 

    "wind_-5.5e-4_new"

    "wind_-5.5e-4_heating_-3.5e-8"
    "wind_-1.5e-4_heating_-3.5e-8"
]

momentum_fluxes_training = [data[1] for data in training_data]
buoyancy_fluxes_training = [data[2] for data in training_data]

momentum_fluxes_interpolating = [data[1] for data in interpolating_data]
buoyancy_fluxes_interpolating = [data[2] for data in interpolating_data]

momentum_fluxes_extrapolating = [data[1] for data in extrapolating_data]
buoyancy_fluxes_extrapolating = [data[2] for data in extrapolating_data]

momentum_fluxes_extrapolating_diurnal = [data[1] for data in extrapolating_data_diurnal]
buoyancy_fluxes_extrapolating_diurnal = [data[2] for data in extrapolating_data_diurnal]

fig = CairoMakie.Figure(resolution=(1000, 650))

ax =fig[1,1] = CairoMakie.Axis(fig, xlabel="Buoyancy Flux / m² s⁻³", ylabel="Momentum Flux / m² s⁻²")
color_palette = distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

rectangle = CairoMakie.poly!(ax, Point2f0[(-5e-8, -2e-4), (-5e-8, -5e-4), (5e-8, -5e-4), (5e-8, -2e-4)], color=("paleturquoise3", 0.5))

training_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_training[i], buoyancy_fluxes_training[i]], 
                    [momentum_fluxes_training[i], momentum_fluxes_training[i]], color=color_palette[1]) for i in 1:length(training_data)]

interpolating_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_interpolating[i], buoyancy_fluxes_interpolating[i]], 
                    [momentum_fluxes_interpolating[i], momentum_fluxes_interpolating[i]], color=color_palette[2]) for i in 1:length(interpolating_data)]

extrapolating_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_extrapolating_diurnal[i], buoyancy_fluxes_extrapolating_diurnal[i]], 
                    [momentum_fluxes_extrapolating_diurnal[i], momentum_fluxes_extrapolating_diurnal[i]], color=color_palette[3]) for i in 1:length(extrapolating_data_diurnal)]

extrapolating_points = CairoMakie.scatter!(ax, buoyancy_fluxes_extrapolating, momentum_fluxes_extrapolating, color=color_palette[3])

# diurnal_line_1 = CairoMakie.lines!(ax, [-5.5e-8, 5.5e-8], [-1.5e-4, -1.5e-4], color=color_palette[4])
# diurnal_line_2 = CairoMakie.lines!(ax, [-5.5e-8, 5.5e-8], [-5.5e-4, -5.5e-4], color=color_palette[4])

# # diurnal_points = CairoMakie.scatter!(ax, buoyancy_fluxes_diurnal, momentum_fluxes_diurnal, color=color_palette[4])

legend = fig[2,1] = CairoMakie.Legend(fig, [training_lines[1], interpolating_lines[1], extrapolating_lines[1], extrapolating_points, rectangle],
                                            ["Training", "Interpolating", "Extrapolating", "Constant Fluxes", "Interpolation Region"], orientation=:horizontal)

rowsize!(fig.layout, 1, CairoMakie.Relative(0.95))
trim!(fig.layout)
fig
save("final_results/data_diurnal.png", fig, px_per_unit = 4)

losses_training = []

for train_file in train_files
    file = jldopen("final_results/3sim_diurnal/train_$(train_file)/profiles_fluxes_oceananigans.jld2")
    data = file["NDE_profile"]
    close(file)

    loss = data["loss"] + data["loss_gradient"]
    loss_mpp = data["loss_modified_pacanowski_philander"] + data["loss_modified_pacanowski_philander_gradient"]
    loss_kpp = data["loss_kpp"] + data["loss_kpp_gradient"]

    loss_min = argmin([loss, loss_mpp, loss_kpp])

    loss_str = ["NDE", "mpp", "kpp"]
    push!(losses_training, loss_str[loss_min])
end

losses_training

losses_interpolating = []

for interpolating_file in interpolating_files
    file = jldopen("final_results/3sim_diurnal/test_$(interpolating_file)/profiles_fluxes_oceananigans.jld2")
    data = file["NDE_profile"]
    close(file)

    loss = data["loss"] + data["loss_gradient"]
    loss_mpp = data["loss_modified_pacanowski_philander"] + data["loss_modified_pacanowski_philander_gradient"]
    loss_kpp = data["loss_kpp"] + data["loss_kpp_gradient"]

    @show loss, loss_mpp, loss_kpp

    loss_min = argmin([loss, loss_mpp, loss_kpp])

    loss_str = ["NDE", "mpp", "kpp"]
    push!(losses_interpolating, loss_str[loss_min])
end

losses_interpolating

losses_extrapolating = []

for extrapolating_file in extrapolating_files
    file = jldopen("final_results/3sim_diurnal/test_$(extrapolating_file)/profiles_fluxes_oceananigans.jld2")
    data = file["NDE_profile"]
    close(file)

    loss = data["loss"] + data["loss_gradient"]
    loss_mpp = data["loss_modified_pacanowski_philander"] + data["loss_modified_pacanowski_philander_gradient"]
    loss_kpp = data["loss_kpp"] + data["loss_kpp_gradient"]

    @show loss, loss_mpp, loss_kpp

    loss_min = argmin([loss, loss_mpp, loss_kpp])

    loss_str = ["NDE", "mpp", "kpp"]
    push!(losses_extrapolating, loss_str[loss_min])
end

losses_extrapolating

losses_extrapolating_diurnal = []

for diurnal_file in extrapolating_diurnal_files
    file = jldopen("final_results/3sim_diurnal/test_$(diurnal_file)/profiles_fluxes_oceananigans.jld2")
    data = file["NDE_profile"]
    close(file)

    loss = data["loss"] + data["loss_gradient"]
    loss_mpp = data["loss_modified_pacanowski_philander"] + data["loss_modified_pacanowski_philander_gradient"]
    loss_kpp = data["loss_kpp"] + data["loss_kpp_gradient"]

    @show loss, loss_mpp, loss_kpp

    loss_min = argmin([loss, loss_mpp, loss_kpp])

    loss_str = ["NDE", "mpp", "kpp"]
    push!(losses_extrapolating_diurnal, loss_str[loss_min])
end

losses_extrapolating_diurnal

fig = CairoMakie.Figure(resolution=(1000, 650))

ax =fig[1,1] = CairoMakie.Axis(fig, xlabel="Buoyancy Flux / m² s⁻³", ylabel="Momentum Flux / m² s⁻²")
color_palette = distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

loss_colors = Dict(
    "NDE" => color_palette[1],
    "mpp" => color_palette[3],
    "kpp" => color_palette[4],
)

rectangle = CairoMakie.poly!(ax, Point2f0[(-5e-8, -2e-4), (-5e-8, -5e-4-3f-6), (5e-8, -5e-4-3f-6), (5e-8, -2e-4)], color=("paleturquoise3", 0.5))

NDE_line = CairoMakie.lines!(ax, [buoyancy_fluxes_training[1], buoyancy_fluxes_training[1]], [momentum_fluxes_training[1], momentum_fluxes_training[1]], color=loss_colors["NDE"])
mpp_line = CairoMakie.lines!(ax, [buoyancy_fluxes_training[1], buoyancy_fluxes_training[1]], [momentum_fluxes_training[1], momentum_fluxes_training[1]], color=loss_colors["mpp"])
kpp_line = CairoMakie.lines!(ax, [buoyancy_fluxes_training[1], buoyancy_fluxes_training[1]], [momentum_fluxes_training[1], momentum_fluxes_training[1]], color=loss_colors["kpp"])

training_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_training[i], buoyancy_fluxes_training[i]], 
                    [momentum_fluxes_training[i], momentum_fluxes_training[i]], color=loss_colors[losses_training[i]]) for i in 1:length(losses_training)]

interpolating_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_interpolating[i], buoyancy_fluxes_interpolating[i]], 
                    [momentum_fluxes_interpolating[i], momentum_fluxes_interpolating[i]], color=loss_colors[losses_interpolating[i]]) for i in 1:length(losses_interpolating)]

extrapolating_lines = [CairoMakie.lines!(ax, [-buoyancy_fluxes_extrapolating[i], buoyancy_fluxes_extrapolating[i]], 
                    [momentum_fluxes_extrapolating[i], momentum_fluxes_extrapolating[i]], color=loss_colors[losses_extrapolating[i]]) for i in 1:length(losses_extrapolating_diurnal)]

extrapolating_points = [CairoMakie.scatter!(ax, [buoyancy_fluxes_extrapolating[i]], [momentum_fluxes_extrapolating[i]], color=loss_colors[losses_extrapolating[i]]) for i in 1:length(losses_extrapolating)]



# training_points = [CairoMakie.scatter!(ax, [buoyancy_fluxes_training[i]], [momentum_fluxes_training[i]], color=loss_colors[losses_training[i]]) for i in 1:length(losses_training)]
# interpolating_points = [CairoMakie.scatter!(ax, [buoyancy_fluxes_interpolating[i]], [momentum_fluxes_interpolating[i]], color=loss_colors[losses_interpolating[i]]) for i in 1:length(losses_interpolating)]
# extrapolating_points = [CairoMakie.scatter!(ax, [buoyancy_fluxes_extrapolating[i]], [momentum_fluxes_extrapolating[i]], color=loss_colors[losses_extrapolating[i]]) for i in 1:length(losses_extrapolating)]

# # extrapolating_points = CairoMakie.scatter!(ax, buoyancy_fluxes_extrapolating, momentum_fluxes_extrapolating, color=color_palette[3])

# diurnal_line_1 = CairoMakie.lines!(ax, [-5.5e-8, 5.5e-8], [-5.5e-4, -5.5e-4], color=loss_colors[losses_diurnal[1]])
# diurnal_line_2 = CairoMakie.lines!(ax, [-5.5e-8, 5.5e-8], [-1.5e-4, -1.5e-4], color=loss_colors[losses_diurnal[2]])

legend = fig[2,1] = CairoMakie.Legend(fig, [mpp_line, kpp_line, NDE_line],
                                ["Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], orientation=:horizontal)

rowsize!(fig.layout, 1, CairoMakie.Relative(0.95))
trim!(fig.layout)
fig
save("final_results/data_diurnal_loss_results.png", fig, px_per_unit = 4)