using Printf
using JLD2
using DataDeps
using OceanTurb
using CairoMakie
using WindMixing
using OceanParameterizations
using Images, FileIO
using ImageTransformations

train_files = [
    # "wind_-5e-4_cooling_3e-8_new",   
    # "wind_-5e-4_cooling_1e-8_new",   
    # "wind_-2e-4_cooling_3e-8_new",   
    # "wind_-2e-4_cooling_1e-8_new",   
    # "wind_-5e-4_heating_-3e-8_new",  
    # "wind_-2e-4_heating_-1e-8_new",  
    # "wind_-2e-4_heating_-3e-8_new",  
    # "wind_-5e-4_heating_-1e-8_new",  
  
    # "wind_-3.5e-4_cooling_2e-8_new", 
    # "wind_-3.5e-4_heating_-2e-8_new",
  
    # "wind_-5e-4_cooling_2e-8_new",   
    # "wind_-3.5e-4_cooling_3e-8_new", 
    # "wind_-3.5e-4_cooling_1e-8_new", 
    # "wind_-2e-4_cooling_2e-8_new",   
    # "wind_-3.5e-4_heating_-3e-8_new",
    # "wind_-3.5e-4_heating_-1e-8_new",
    # "wind_-2e-4_heating_-2e-8_new",  
    # "wind_-5e-4_heating_-2e-8_new",

    # testing
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
    "wind_-5.5e-4_diurnal_5.5e-8", 
    "wind_-1.5e-4_diurnal_5.5e-8", 
    "wind_-2e-4_diurnal_4e-8",    
  
    "wind_-5.5e-4_new",            
  
    "wind_-5.5e-4_heating_-3.5e-8",
    "wind_-1.5e-4_heating_-3.5e-8",
    "wind_-5.5e-4_cooling_3.5e-8", 
    "wind_-1.5e-4_cooling_3.5e-8", 
]

FILE_DIR = "Data/kpp_eki_uvT_180ensemble_1000iters_18sim2"
mkpath(FILE_DIR)

for train_file in train_files
    @info train_file
    𝒟 = WindMixing.data([train_file], scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

    DATA_PATH = "Data/kpp_eki_uvT_180ensemble_1000iters_18sim_timeseries.jld2"

    file = jldopen(DATA_PATH)
    u = file["$(train_file)/u"]
    v = file["$(train_file)/v"]
    T = file["$(train_file)/T"]
    uw = file["$(train_file)/uw"]
    vw = file["$(train_file)/vw"]
    wT = file["$(train_file)/wT"]

    close(file)

    times_days = 𝒟.t ./ 86400

    frame = Node(1)

    u_frame_kpp = @lift u[:,$frame]
    v_frame_kpp = @lift v[:,$frame]
    T_frame_kpp = @lift T[:,$frame]

    uw_frame_kpp = @lift uw[:,$frame]
    vw_frame_kpp = @lift vw[:,$frame]
    wT_frame_kpp = @lift wT[:,$frame]

    u_frame_truth = @lift 𝒟.u.coarse[:,$frame]
    v_frame_truth = @lift 𝒟.v.coarse[:,$frame]
    T_frame_truth = @lift 𝒟.T.coarse[:,$frame]

    uw_frame_truth = @lift 𝒟.uw.coarse[:,$frame]
    vw_frame_truth = @lift 𝒟.vw.coarse[:,$frame]
    wT_frame_truth = @lift 𝒟.wT.coarse[:,$frame]

    @inline function find_lims(dataset₁, dataset₂)
        datasets = [dataset₁, dataset₂]
        return maximum(maximum.(datasets)), minimum(minimum.(datasets))
    end

    u_max, u_min = find_lims(u, 𝒟.u.coarse)
    v_max, v_min = find_lims(v, 𝒟.v.coarse)
    T_max, T_min = find_lims(T, 𝒟.T.coarse)

    uw_max, uw_min = find_lims(uw, 𝒟.uw.coarse)
    vw_max, vw_min = find_lims(vw, 𝒟.vw.coarse)
    wT_max, wT_min = find_lims(wT, 𝒟.wT.coarse)

    BC_str = @sprintf "Momentum Flux = %.1e m² s⁻², Temperature Flux = %.1e m s⁻¹ °C" uw[end, 1] maximum(wT[end, :])
    plot_title = @lift "$BC_str, Time = $(round(times_days[$frame], digits=2)) days"

    # fig = Figure(resolution=(1920, 1080))
    fig = Figure(resolution=(1920, 1080))

    zc = 𝒟.u.z
    zf = 𝒟.uw.z
    zf_interior = zf[2:end-1]

    ax_u = fig[1, 1] = CairoMakie.Axis(fig, xlabel="u (m s⁻¹)", ylabel="z (m)")
    ax_v = fig[1, 2] = CairoMakie.Axis(fig, xlabel="v (m s⁻¹)", ylabel="z (m)")
    ax_T = fig[1, 3] = CairoMakie.Axis(fig, xlabel="T (m s⁻¹)", ylabel="z (m)")
    ax_uw = fig[2, 1] = CairoMakie.Axis(fig, xlabel="uw (m² s⁻²)", ylabel="z (m)")
    ax_vw = fig[2, 2] = CairoMakie.Axis(fig, xlabel="vw (m² s⁻²)", ylabel="z (m)")
    ax_wT = fig[2, 3] = CairoMakie.Axis(fig, xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")

    alpha=0.5
    truth_linewidth = 7
    linewidth = 3

    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    u_line_kpp = lines!(ax_u, u_frame_kpp, zc)
    v_line_kpp = lines!(ax_v, v_frame_kpp, zc)
    T_line_kpp = lines!(ax_T, T_frame_kpp, zc)

    uw_line_kpp = lines!(ax_uw, uw_frame_kpp, zf)
    vw_line_kpp = lines!(ax_vw, vw_frame_kpp, zf)
    wT_line_kpp = lines!(ax_wT, wT_frame_kpp, zf)

    u_line_truth = lines!(ax_u, u_frame_truth, zc)
    v_line_truth = lines!(ax_v, v_frame_truth, zc)
    T_line_truth = lines!(ax_T, T_frame_truth, zc)

    uw_line_truth = lines!(ax_uw, uw_frame_truth, zf)
    vw_line_truth = lines!(ax_vw, vw_frame_truth, zf)
    wT_line_truth = lines!(ax_wT, wT_frame_truth, zf)

    axislegend(ax_T, [T_line_kpp, T_line_truth], ["KPP", "Oceananigans.jl"], position = :rb)

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=25)

    trim!(fig.layout)

    print_frame = maximum([1, Int(floor(length(times_days)/20))])

    function print_progress(n, n_total, print_frame, type)
        if n % print_frame == 0
            @info "Animating $(type) frame $n/$n_total"
        end
    end

    @info "Starting Animation"
    FILE_PATH = "$(FILE_DIR)/test_$(train_file)"

    CairoMakie.record(fig, "$FILE_PATH.mp4", 1:length(times_days), framerate=60, compression=1) do n
        # print_progress(n, length(times_days), print_frame, "mp4")
        frame[] = n
    end
end