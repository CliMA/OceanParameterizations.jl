using CairoMakie
using Colors
using Makie.GeometryBasics
using JLD2

ρ_ref = 1026
α = 2e-4
cₚ = 3991
g = 9.81

b_to_T = ρ_ref * cₚ / (α * g)

training_data = [
    (5e-4 * ρ_ref, -1e-8 * b_to_T),
    (5e-4 * ρ_ref, -2e-8 * b_to_T),
    (5e-4 * ρ_ref, -3e-8 * b_to_T),

    (3.5e-4 * ρ_ref, -1e-8 * b_to_T),
    (3.5e-4 * ρ_ref, -2e-8 * b_to_T),
    (3.5e-4 * ρ_ref, -3e-8 * b_to_T),

    (2e-4 * ρ_ref, -1e-8 * b_to_T),
    (2e-4 * ρ_ref, -2e-8 * b_to_T),
    (2e-4 * ρ_ref, -3e-8 * b_to_T),

    (5e-4 * ρ_ref, 1e-8 * b_to_T),
    (5e-4 * ρ_ref, 2e-8 * b_to_T),
    (5e-4 * ρ_ref, 3e-8 * b_to_T),

    (3.5e-4 * ρ_ref, 1e-8 * b_to_T),
    (3.5e-4 * ρ_ref, 2e-8 * b_to_T),
    (3.5e-4 * ρ_ref, 3e-8 * b_to_T),

    (2e-4 * ρ_ref, 1e-8 * b_to_T),
    (2e-4 * ρ_ref, 2e-8 * b_to_T),
    (2e-4 * ρ_ref, 3e-8 * b_to_T),
]

train_files = [
    "wind_-5e-4_heating_-1e-8_new"  
    "wind_-5e-4_heating_-2e-8_new"  
    "wind_-5e-4_heating_-3e-8_new"  

    "wind_-3.5e-4_heating_-1e-8_new"
    "wind_-3.5e-4_heating_-2e-8_new"
    "wind_-3.5e-4_heating_-3e-8_new"
    
    "wind_-2e-4_heating_-1e-8_new"  
    "wind_-2e-4_heating_-2e-8_new"  
    "wind_-2e-4_heating_-3e-8_new"  
    
    "wind_-5e-4_cooling_1e-8_new"   
    "wind_-5e-4_cooling_2e-8_new"   
    "wind_-5e-4_cooling_3e-8_new"       

    "wind_-3.5e-4_cooling_1e-8_new" 
    "wind_-3.5e-4_cooling_2e-8_new" 
    "wind_-3.5e-4_cooling_3e-8_new" 

    "wind_-2e-4_cooling_1e-8_new"   
    "wind_-2e-4_cooling_2e-8_new"   
    "wind_-2e-4_cooling_3e-8_new"   
]

interpolating_data = [
    (4.5e-4 * ρ_ref, 2.5e-8 * b_to_T),
    (4.5e-4 * ρ_ref, 1.5e-8 * b_to_T),
    (2.5e-4 * ρ_ref, 2.5e-8 * b_to_T),
    (2.5e-4 * ρ_ref, 1.5e-8 * b_to_T),

    (4.5e-4 * ρ_ref, -2.5e-8 * b_to_T),
    (4.5e-4 * ρ_ref, -1.5e-8 * b_to_T),
    (2.5e-4 * ρ_ref, -2.5e-8 * b_to_T),
    (2.5e-4 * ρ_ref, -1.5e-8 * b_to_T),
]

interpolating_files = [
    "wind_-4.5e-4_cooling_2.5e-8" 
    "wind_-4.5e-4_cooling_1.5e-8" 
    "wind_-2.5e-4_cooling_2.5e-8" 
    "wind_-2.5e-4_cooling_1.5e-8" 

    "wind_-4.5e-4_heating_-2.5e-8"
    "wind_-4.5e-4_heating_-1.5e-8"
    "wind_-2.5e-4_heating_-2.5e-8"
    "wind_-2.5e-4_heating_-1.5e-8"
]

extrapolating_data = [
    (1.5e-4 * ρ_ref, 3.5e-8 * b_to_T),
    (5.5e-4 * ρ_ref, 3.5e-8 * b_to_T),
    (5.5e-4 * ρ_ref, 0 * b_to_T),
    (5.5e-4 * ρ_ref, -3.5e-8 * b_to_T),
    (1.5e-4 * ρ_ref, -3.5e-8 * b_to_T),
]

extrapolating_files = [
    "wind_-1.5e-4_cooling_3.5e-8" 
    "wind_-5.5e-4_cooling_3.5e-8" 

    "wind_-5.5e-4_new"

    "wind_-5.5e-4_heating_-3.5e-8"
    "wind_-1.5e-4_heating_-3.5e-8"
]

diurnal_data = [
    (5.5e-4 * ρ_ref, 5.5e-8 * b_to_T),
    (1.5e-4 * ρ_ref, 5.5e-8 * b_to_T),
]

diurnal_files = [
    "wind_-5.5e-4_diurnal_5.5e-8"
    "wind_-1.5e-4_diurnal_5.5e-8"
]

momentum_fluxes_training = [data[1] for data in training_data]
buoyancy_fluxes_training = [data[2] for data in training_data]

momentum_fluxes_interpolating = [data[1] for data in interpolating_data]
buoyancy_fluxes_interpolating = [data[2] for data in interpolating_data]

momentum_fluxes_extrapolating = [data[1] for data in extrapolating_data]
buoyancy_fluxes_extrapolating = [data[2] for data in extrapolating_data]

momentum_fluxes_diurnal = [data[1] for data in diurnal_data]
buoyancy_fluxes_diurnal = [data[2] for data in diurnal_data]

##
fig = Figure(resolution=(1000, 800), fontsize=25)

ax =fig[1,1] = Axis(fig, xlabel=L"Heat flux (W m$^{-2}$)", ylabel=L"Wind stress / N m$^{-2}$")
color_palette = distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)


# rectangle = poly!(ax, Point2f0[(-3e-8, -2e-4), (-3e-8, -5e-4), (3e-8, -5e-4), (3e-8, -2e-4)], color=("paleturquoise3", 0.5))
rectangle = poly!(ax, Point2f[(-3e-8 * b_to_T, 2e-4 * ρ_ref), (-3e-8 * b_to_T, 5e-4 * ρ_ref), (3e-8 * b_to_T, 5e-4 * ρ_ref), (3e-8 * b_to_T, 2e-4 * ρ_ref)], color=("paleturquoise3", 0.5))

training_points = scatter!(ax, buoyancy_fluxes_training, momentum_fluxes_training, color=color_palette[1], markersize=20, marker=:circle)
interpolating_points = scatter!(ax, buoyancy_fluxes_interpolating, momentum_fluxes_interpolating, color=color_palette[2], markersize=20, marker=:cross)
extrapolating_points = scatter!(ax, buoyancy_fluxes_extrapolating, momentum_fluxes_extrapolating, color=color_palette[3], markersize=20, marker=:diamond)

diurnal_line_1 = arrows!(ax, [0, 0], [1.5e-4 * ρ_ref, 1.5e-4 * ρ_ref], [5.5e-8 * b_to_T, -5.5e-8 * b_to_T], [0], linewidth=3, arrowsize=20, color=color_palette[4])
diurnal_line_2 = arrows!(ax, [0, 0], [5.5e-4 * ρ_ref, 5.5e-4 * ρ_ref], [5.5e-8 * b_to_T, -5.5e-8 * b_to_T], [0], linewidth=3, arrowsize=20, color=color_palette[4])

hidden_line_3 = lines!(ax, [-5.5e-8 * b_to_T, 5.5e-8 * b_to_T], [5.5e-4 * ρ_ref, 5.5e-4 * ρ_ref], color=color_palette[4], linewidth=4)

legend = fig[2,1] = Legend(fig, [training_points, interpolating_points, extrapolating_points, hidden_line_3, rectangle],
                           ["Training", "Interpolating", "Extrapolating", "Diurnal Fluxes", "Interpolation Region"], 
                           orientation=:horizontal,
                           nbanks=2)

display(fig)
save("plots/training_regions.pdf", fig, px_per_unit=4, pt_per_unit=4)
##