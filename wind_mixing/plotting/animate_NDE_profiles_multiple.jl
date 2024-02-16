using CairoMakie
using FileIO
using JLD2
using Colors
using Printf

NDE_DIR = "NDE_18sim_windcooling_windheating_18simLBFGST0.8nogradnonlocal_divide1f5_gradient_smallNN_leakyrelu_layers_1_rate_2e-4_T0.8_2e-4_300iter"

train_files_all = [
  "wind_-5e-4_cooling_3e-8_new",   
  "wind_-5e-4_cooling_1e-8_new",   
  "wind_-2e-4_cooling_3e-8_new",   
  "wind_-2e-4_cooling_1e-8_new",   
  "wind_-5e-4_heating_-3e-8_new",  
  "wind_-2e-4_heating_-1e-8_new",  
  "wind_-2e-4_heating_-3e-8_new",  
  "wind_-5e-4_heating_-1e-8_new",  

  "wind_-3.5e-4_cooling_2e-8_new", 
  "wind_-3.5e-4_heating_-2e-8_new",

  "wind_-5e-4_cooling_2e-8_new",   
  "wind_-3.5e-4_cooling_3e-8_new", 
  "wind_-3.5e-4_cooling_1e-8_new", 
  "wind_-2e-4_cooling_2e-8_new",   
  "wind_-3.5e-4_heating_-3e-8_new",
  "wind_-3.5e-4_heating_-1e-8_new",
  "wind_-2e-4_heating_-2e-8_new",  
  "wind_-5e-4_heating_-2e-8_new",
]

test_files = [
  "wind_-5e-4_cooling_3e-8_new",   
  "wind_-5e-4_cooling_1e-8_new",   
  "wind_-2e-4_cooling_3e-8_new",   
  "wind_-2e-4_cooling_1e-8_new",   
  "wind_-5e-4_heating_-3e-8_new",  
  "wind_-2e-4_heating_-1e-8_new",  
  "wind_-2e-4_heating_-3e-8_new",  
  "wind_-5e-4_heating_-1e-8_new",  

  "wind_-3.5e-4_cooling_2e-8_new", 
  "wind_-3.5e-4_heating_-2e-8_new",

  "wind_-5e-4_cooling_2e-8_new",   
  "wind_-3.5e-4_cooling_3e-8_new", 
  "wind_-3.5e-4_cooling_1e-8_new", 
  "wind_-2e-4_cooling_2e-8_new",   
  "wind_-3.5e-4_heating_-3e-8_new",
  "wind_-3.5e-4_heating_-1e-8_new",
  "wind_-2e-4_heating_-2e-8_new",  
  "wind_-5e-4_heating_-2e-8_new",

  "wind_-5e-4_diurnal_5e-8",    
  "wind_-5e-4_diurnal_3e-8",    
  "wind_-5e-4_diurnal_1e-8",    
     
  "wind_-3.5e-4_diurnal_5e-8",  
  "wind_-3.5e-4_diurnal_3e-8",  
  "wind_-3.5e-4_diurnal_1e-8",  
     
  "wind_-2e-4_diurnal_5e-8",    
  "wind_-2e-4_diurnal_3e-8",    

  "wind_-2e-4_diurnal_1e-8",    
     
  "wind_-2e-4_diurnal_2e-8",    
  "wind_-2e-4_diurnal_3.5e-8", 
  "wind_-3.5e-4_diurnal_2e-8",
  "wind_-3.5e-4_diurnal_3.5e-8",
  "wind_-5e-4_diurnal_2e-8",    
  "wind_-5e-4_diurnal_3.5e-8",
  
  "cooling_5e-8_new",            
  "cooling_4.5e-8_new",          
  "cooling_4e-8_new",            
  "cooling_3.5e-8_new",         
  "cooling_3e-8_new",            
  "cooling_2.5e-8_new",          
  "cooling_2e-8_new",            
  "cooling_1.5e-8_new",          
  "cooling_1e-8_new",   

  "wind_-5e-4_new",              
  "wind_-4.5e-4_new",            
  "wind_-4e-4_new",              
  "wind_-3.5e-4_new",            
  "wind_-3e-4_new",              
  "wind_-2.5e-4_new",            
  "wind_-2e-4_new",       

  "wind_-4.5e-4_cooling_2.5e-8",
  "wind_-2.5e-4_cooling_1.5e-8", 
  "wind_-4.5e-4_cooling_1.5e-8", 
  "wind_-2.5e-4_cooling_2.5e-8", 

  "wind_-4.5e-4_heating_-2.5e-8",
  "wind_-2.5e-4_heating_-1.5e-8",
  "wind_-4.5e-4_heating_-1.5e-8",
  "wind_-2.5e-4_heating_-2.5e-8",

  "wind_-4.5e-4_diurnal_4e-8",   
  "wind_-4.5e-4_diurnal_2e-8",   
  "wind_-3e-4_diurnal_4e-8",     
  "wind_-3e-4_diurnal_2e-8",     
  "wind_-2e-4_diurnal_2e-8",     
  "wind_-5.5e-4_diurnal_5.5e-8", 
  "wind_-1.5e-4_diurnal_5.5e-8", 
  "wind_-2e-4_diurnal_4e-8",    

  "wind_-5.5e-4_new",            

  "wind_-5.5e-4_heating_-3.5e-8",
  "wind_-1.5e-4_heating_-3.5e-8",
  "wind_-5.5e-4_cooling_3.5e-8", 
  "wind_-1.5e-4_cooling_3.5e-8", 
]

train_filenames = ["wind_-5e-4_cooling_1e-8_new", "wind_-2e-4_cooling_3e-8_new"]
interpolate_filenames = ["wind_-4.5e-4_heating_-2.5e-8", "wind_-2.5e-4_cooling_1.5e-8"]
extrapolate_filenames = ["wind_-5.5e-4_new", "wind_-1.5e-4_heating_-3.5e-8"]
diurnal_filenames = ["wind_-4.5e-4_diurnal_4e-8", "wind_-2e-4_diurnal_2e-8"]

f_kpp = jldopen("../Data/kpp_eki_180ensemble_1000iters_18sim_timeseries.jld2")

kpp_profiles_train = [f_kpp[filename] for filename in train_filenames]
kpp_profiles_interpolate = [f_kpp[filename] for filename in interpolate_filenames]
kpp_profiles_extrapolate = [f_kpp[filename] for filename in extrapolate_filenames]
kpp_profiles_diurnal = [f_kpp[filename] for filename in diurnal_filenames]

train_data = [jldopen(joinpath("..", "Output", NDE_DIR, "train_$(filename)", "solution_oceananigans.jld2"))["NDE_profile"] for filename in train_filenames]
interpolate_data = [jldopen(joinpath("..", "Output", NDE_DIR, "test_$(filename)", "solution_oceananigans.jld2"))["NDE_profile"] for filename in interpolate_filenames]
extrapolate_data = [jldopen(joinpath("..", "Output", NDE_DIR, "test_$(filename)", "solution_oceananigans.jld2"))["NDE_profile"] for filename in extrapolate_filenames]
diurnal_data = [jldopen(joinpath("..", "Output", NDE_DIR, "test_$(filename)", "solution_oceananigans.jld2"))["NDE_profile"] for filename in diurnal_filenames]

T_data_train = [[data["truth_T"], data["test_T_modified_pacanowski_philander"], kpp_profiles["T"], data["test_T"]] for (data, kpp_profiles) in zip(train_data, kpp_profiles_train)]
T_data_interpolate = [[data["truth_T"], data["test_T_modified_pacanowski_philander"], kpp_profiles["T"], data["test_T"]] for (data, kpp_profiles) in zip(interpolate_data, kpp_profiles_interpolate)]
T_data_extrapolate = [[data["truth_T"], data["test_T_modified_pacanowski_philander"], kpp_profiles["T"], data["test_T"]] for (data, kpp_profiles) in zip(extrapolate_data, kpp_profiles_extrapolate)]
T_data_diurnal = [[data["truth_T"], data["test_T_modified_pacanowski_philander"], kpp_profiles["T"], data["test_T"]] for (data, kpp_profiles) in zip(diurnal_data, kpp_profiles_diurnal)]

times = train_data[1]["t"] ./ 86400

@inline function find_lims(datasets)
    return minimum(minimum.(datasets)), maximum(maximum.(datasets))
end

T_train_lims = [find_lims(T_data) for T_data in T_data_train]
T_interpolate_lims = [find_lims(T_data) for T_data in T_data_interpolate]
T_extrapolate_lims = [find_lims(T_data) for T_data in T_data_extrapolate]
T_diurnal_lims = [find_lims(T_data) for T_data in T_data_diurnal]

#%%
frame = Observable(1)

T_trainₙ = [[@lift data[:,$frame] for data in T_data] for T_data in T_data_train]
T_interpolateₙ = [[@lift data[:,$frame] for data in T_data] for T_data in T_data_interpolate]
T_extrapolateₙ = [[@lift data[:,$frame] for data in T_data] for T_data in T_data_extrapolate]
T_diurnalₙ = [[@lift data[:,$frame] for data in T_data] for T_data in T_data_diurnal]

fig = Figure(resolution=(1920, 1080), fontsize=25)

colors = distinguishable_colors(length(T_data_train[1]), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

zc = train_data[1]["depth_profile"]

axtrain₁ = fig[2, 1] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Strong Wind Weak Cooling", titlefont=:regular)
axtrain₂ = fig[2, 2] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Weak Wind Strong Cooling", titlefont=:regular)
axinterpolate₁ = fig[2, 3] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Strong Wind Strong Heating", titlefont=:regular)
axinterpolate₂ = fig[2, 4] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Weak Wind Weak Cooling", titlefont=:regular)
axextrapolate₁ = fig[4, 1] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Strong Wind No Heat Flux", titlefont=:regular)
axextrapolate₂ = fig[4, 2] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Weak Wind Strong Heating", titlefont=:regular)
axdiurnal₁ = fig[4, 3] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Strong Wind Strong Cycle", titlefont=:regular)
axdiurnal₂ = fig[4, 4] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $", title="Weak Wind Weak Cycle", titlefont=:regular)

Label(fig[1, 1:2], "Training", tellwidth=false, font=:bold)
Label(fig[1, 3:4], "Interpolation", tellwidth=false, font=:bold)
Label(fig[3, 1:2], "Extrapolation", tellwidth=false, font=:bold)
Label(fig[3, 3:4], "Diurnal", tellwidth=false, font=:bold)

CairoMakie.xlims!(axtrain₁, T_train_lims[1])
CairoMakie.xlims!(axtrain₂, T_train_lims[2])
CairoMakie.xlims!(axinterpolate₁, T_interpolate_lims[1])
CairoMakie.xlims!(axinterpolate₂, T_interpolate_lims[2])
CairoMakie.xlims!(axextrapolate₁, T_extrapolate_lims[1])
CairoMakie.xlims!(axextrapolate₂, T_extrapolate_lims[2])
CairoMakie.xlims!(axdiurnal₁, T_diurnal_lims[1])
CairoMakie.xlims!(axdiurnal₂, T_diurnal_lims[2])

CairoMakie.ylims!(axtrain₁, minimum(zc), 0)
CairoMakie.ylims!(axtrain₂, minimum(zc), 0)
CairoMakie.ylims!(axinterpolate₁, minimum(zc), 0)
CairoMakie.ylims!(axinterpolate₂, minimum(zc), 0)
CairoMakie.ylims!(axextrapolate₁, minimum(zc), 0)
CairoMakie.ylims!(axextrapolate₂, minimum(zc), 0)
CairoMakie.ylims!(axdiurnal₁, minimum(zc), 0)
CairoMakie.ylims!(axdiurnal₂, minimum(zc), 0)

linkyaxes!(axtrain₁, axtrain₂, axinterpolate₁, axinterpolate₂, axextrapolate₁, axextrapolate₂, axdiurnal₁, axdiurnal₂)

# linkxaxes!(axtrain₁, axtrain₂, axinterpolate₁, axinterpolate₂, axextrapolate₁, axextrapolate₂, axdiurnal₁, axdiurnal₂)

hideydecorations!(axtrain₁, ticks=false, ticklabels=false, label=false)
hideydecorations!(axtrain₂)
hideydecorations!(axinterpolate₁)
hideydecorations!(axinterpolate₂)
hideydecorations!(axextrapolate₁, ticks=false, ticklabels=false, label=false)
hideydecorations!(axextrapolate₂)
hideydecorations!(axdiurnal₁)
hideydecorations!(axdiurnal₂)

hidexdecorations!(axtrain₁, ticks=false, ticklabels=false)
hidexdecorations!(axtrain₂, ticks=false, ticklabels=false)
hidexdecorations!(axinterpolate₁, ticks=false, ticklabels=false)
hidexdecorations!(axinterpolate₂, ticks=false, ticklabels=false)
hidexdecorations!(axextrapolate₁, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axextrapolate₂, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axdiurnal₁, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axdiurnal₂, ticks=false, ticklabels=false, label=false)

truth_linewidth = 12
truth_alpha = 0.4
lines!(axtrain₁, T_trainₙ[1][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, label="Large Eddy Simulation", color=colors[1])
lines!(axtrain₂, T_trainₙ[2][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axinterpolate₁, T_interpolateₙ[1][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axinterpolate₂, T_interpolateₙ[2][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axextrapolate₁, T_extrapolateₙ[1][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axextrapolate₂, T_extrapolateₙ[2][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axdiurnal₁, T_diurnalₙ[1][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])
lines!(axdiurnal₂, T_diurnalₙ[2][1], zc, linewidth=truth_linewidth, alpha=truth_alpha, color=colors[1])

data_linewidth = 5
lines!(axtrain₁, T_trainₙ[1][2], zc, linewidth=data_linewidth, color=colors[2], label="Ri-based Diffusivity Only")
lines!(axtrain₂, T_trainₙ[2][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axinterpolate₁, T_interpolateₙ[1][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axinterpolate₂, T_interpolateₙ[2][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axextrapolate₁, T_extrapolateₙ[1][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axextrapolate₂, T_extrapolateₙ[2][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axdiurnal₁, T_diurnalₙ[1][2], zc, linewidth=data_linewidth, color=colors[2])
lines!(axdiurnal₂, T_diurnalₙ[2][2], zc, linewidth=data_linewidth, color=colors[2])

lines!(axtrain₁, T_trainₙ[1][3], zc, linewidth=data_linewidth, color=colors[3], label="K-Profile Parameterisation")
lines!(axtrain₂, T_trainₙ[2][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axinterpolate₁, T_interpolateₙ[1][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axinterpolate₂, T_interpolateₙ[2][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axextrapolate₁, T_extrapolateₙ[1][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axextrapolate₂, T_extrapolateₙ[2][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axdiurnal₁, T_diurnalₙ[1][3], zc, linewidth=data_linewidth, color=colors[3])
lines!(axdiurnal₂, T_diurnalₙ[2][3], zc, linewidth=data_linewidth, color=colors[3])

lines!(axtrain₁, T_trainₙ[1][4], zc, linewidth=data_linewidth, color=colors[4], label="NN Embedded in Oceananigans.jl")
lines!(axtrain₂, T_trainₙ[2][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axinterpolate₁, T_interpolateₙ[1][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axinterpolate₂, T_interpolateₙ[2][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axextrapolate₁, T_extrapolateₙ[1][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axextrapolate₂, T_extrapolateₙ[2][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axdiurnal₁, T_diurnalₙ[1][4], zc, linewidth=data_linewidth, color=colors[4])
lines!(axdiurnal₂, T_diurnalₙ[2][4], zc, linewidth=data_linewidth, color=colors[4])

Legend(fig[5, :], axtrain₁, orientation=:horizontal)

plot_title = @lift "Time = $(round(times[$frame], digits=1)) days"
Label(fig[0, :], plot_title, tellwidth=false, font=:bold)

record(fig, joinpath(".", "plots", "T_profiles.mp4"), 1:length(times), framerate=30, compression=1) do n
    frame[] = n
end
# display(fig)
#%%
for filename in test_files
    @info filename

    if filename in train_files_all
        FILE_DIR = "train_$filename"
    else
        FILE_DIR = "test_$filename"
    end

    file = jldopen(joinpath("..", "Output", NDE_DIR, FILE_DIR, "solution_oceananigans.jld2"))
    data = file["NDE_profile"]
    close(file)

    kpp_profiles = f_kpp[filename]
    times = data["t"] ./ 86400

    # close(f_kpp)

    ## TODO: Run Kpp fluxes
    u_data = [
        data["truth_u"],
        data["test_u_modified_pacanowski_philander"],
        kpp_profiles["u"],
        data["test_u"],
    ]

    v_data = [
        data["truth_v"],
        data["test_v_modified_pacanowski_philander"],
        kpp_profiles["v"],
        data["test_v"],
    ]

    T_data = [
        data["truth_T"],
        data["test_T_modified_pacanowski_philander"],
        kpp_profiles["T"],
        data["test_T"],
    ]

    uw_data = [
        data["truth_uw"],
        data["test_uw_modified_pacanowski_philander"],
        data["test_uw_kpp"],
        # remember to change!!!
        data["test_uw"],
    ]

    vw_data = [
        data["truth_vw"],
        data["test_vw_modified_pacanowski_philander"],
        data["test_vw_kpp"],
        # remember to change!!!
        data["test_vw"],
    ]

    wT_data = [
        data["truth_wT"],
        data["test_wT_modified_pacanowski_philander"],
        data["test_wT_kpp"],
        # remember to change!!!
        data["test_wT"],
    ]

    uw_data .*= 1f4
    vw_data .*= 1f4
    wT_data .*= 1f5

    Ri_data = [
        clamp.(data["truth_Ri"], -1, 2),
        clamp.(data["test_Ri_modified_pacanowski_philander"], -1, 2),
        clamp.(data["test_Ri_kpp"], -1, 2),
        clamp.(data["test_Ri"], -1, 2),
    ]

    close(f_kpp)
    # losses_point_frames = [@lift [data[$frame]] for data in losses_data]

    @inline function find_lims(datasets)
        return maximum(maximum.(datasets)), minimum(minimum.(datasets))
    end

    u_max, u_min = find_lims(u_data)
    v_max, v_min = find_lims(v_data)
    T_max, T_min = find_lims(T_data)

    uw_max, uw_min = find_lims(uw_data)
    vw_max, vw_min = find_lims(vw_data)
    wT_max, wT_min = find_lims(wT_data)

    ##
    frame = Observable(1)

    u_frames = [@lift data[:,$frame] for data in u_data]
    v_frames = [@lift data[:,$frame] for data in v_data]
    T_frames = [@lift data[:,$frame] for data in T_data]

    uw_frames = [@lift data[:,$frame] for data in uw_data]
    vw_frames = [@lift data[:,$frame] for data in vw_data]
    wT_frames = [@lift data[:,$frame] for data in wT_data]

    Ri_frames = [@lift data[:,$frame] for data in Ri_data]

    ρ_ref = 1026
    α = 2e-4
    cₚ = 3991
    g = 9.81

    b_to_T = ρ_ref * cₚ / (α * g)

    BC_str = @sprintf "Wind Stress = %.1e N m⁻², Heat Flux = %.1e W m⁻²" data["truth_uw"][end, 1]*ρ_ref maximum(data["truth_wT"][end, :])*b_to_T/1f5
    plot_title = @lift "$BC_str, Time = $(round(times[$frame], digits=1)) days"

    fig = Figure(resolution=(1920, 960), fontsize=30)

    colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    temp_color = colors[2]
    colors[2] = colors[4]
    colors[4] = temp_color

    zc = data["depth_profile"]
    zf = data["depth_flux"]

    ax_u = fig[1, 1] = Axis(fig, xlabel=L"$\overline{u}$ (m s$^{-1}$)", ylabel=L"z (m) $ $")
    ax_v = fig[1, 2] = Axis(fig, xlabel=L"$\overline{v}$ (m s$^{-1}$)", ylabel=L"z (m) $ $")
    ax_T = fig[1:2, 3:4] = Axis(fig, xlabel=L"$\overline{T}$ ($\degree$C)", ylabel=L"z (m) $ $")
    ax_Ri = fig[1, 5] = Axis(fig, xlabel=L"Ri$ $", ylabel=L"z (m) $ $")
    ax_uw = fig[2, 1] = Axis(fig, xlabel=L"$\overline{u\prime w\prime}$ ($\times 10^{-4}$ m$^2$ s$^{-2}$)", ylabel=L"z (m) $ $")
    ax_vw = fig[2, 2] = Axis(fig, xlabel=L"$\overline{v\prime w\prime}$ ($\times 10^{-4}$ m$^2$ s$^{-2}$)", ylabel=L"z (m) $ $")
    ax_wT = fig[2, 5] = Axis(fig, xlabel=L"$\overline{w\prime T\prime}$ ($\times 10^{-5}$ $\degree$C m s$^{-1}$)", ylabel=L"z (m) $ $")
    title = fig[0, :] = Label(fig, plot_title)

    axs = [ax_u, ax_v, ax_T, ax_Ri, ax_uw, ax_vw, ax_wT]

    alpha=0.4
    truth_linewidth = 12
    linewidth = 5

    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.xlims!(ax_Ri, -1, 2)

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    linkyaxes!(axs...)

    hideydecorations!(ax_v, grid = false)
    hideydecorations!(ax_T, grid = false)
    hideydecorations!(ax_Ri, grid = false)
    hideydecorations!(ax_vw, grid = false)
    hideydecorations!(ax_wT, grid = false)

    # label_a = fig[1, 1, TopLeft()] = Label(fig, "A", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_b = fig[1, 2, TopLeft()] = Label(fig, "B", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_c = fig[1, 3, TopLeft()] = Label(fig, "C", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_d = fig[1, 4, TopLeft()] = Label(fig, "D", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_e = fig[2, 1, TopLeft()] = Label(fig, "E", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_f = fig[2, 2, TopLeft()] = Label(fig, "F", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)
    # label_g = fig[2, 3, TopLeft()] = Label(fig, "G", fontsize = 40, halign = :right, padding = (0, 45, 10, 0), font=:bold)

    u_lines = [
            lines!(ax_u, u_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_u, u_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
    ]

    v_lines = [
            lines!(ax_v, v_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_v, v_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
    ]

    T_lines = [
            # lines!(ax_T, data["truth_T"][:,1], zc, linewidth=linewidth, color=colors[end], linestyle=:dash)
            lines!(ax_T, T_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_T, T_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(T_data)]
    ]

    uw_lines = [
            lines!(ax_uw, uw_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_uw, uw_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(uw_data)]
    ]

    vw_lines = [
            lines!(ax_vw, vw_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_vw, vw_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(vw_data)]
    ]

    wT_lines = [
        lines!(ax_wT, wT_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_wT, wT_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(wT_data)]
    ]

    Ri_lines = [
            lines!(ax_Ri, Ri_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_Ri, Ri_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(Ri_data)]
    ]

    # axislegend(ax_T, T_lines, ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], 
    #        "Data Type", tellwidth=false, orientation=:horizontal, nbanks=2)

    axislegend(ax_T, T_lines, ["Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], 
        "Data Type", position=:rb)
    # display(fig)
    ##
    record(fig, joinpath("..", "Output", NDE_DIR, FILE_DIR, "$FILE_DIR.mp4"), 1:length(times), framerate=30, compression=1) do n
        frame[] = n
    end
end
