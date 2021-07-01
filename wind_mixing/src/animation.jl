function animate_NN(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str, PATH=joinpath(pwd(), "Output"))
    anim = @animate for n in 1:size(xs[1], 2)
        x_max = maximum(maximum(x) for x in xs)
        x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1], 2))"
        fig = Plots.plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            Plots.plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n] / 86400, digits=2)) days")
        end
        Plots.xlabel!(fig, "$x_str")
        Plots.ylabel!(fig, "z")
    end
    # gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
    mp4(anim, joinpath(PATH, "$(filename).mp4"), fps=30)
end

function animate_profile(data, profile_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_profile = data["truth_$profile_type"]
    test_profile = data["test_$profile_type"]

    profile_max = maximum([maximum(truth_profile), maximum(test_profile)])
    profile_min = minimum([minimum(truth_profile), minimum(test_profile)])

    t = data["t"]

    z_profile = data["depth_profile"]

    z_max = maximum(z_profile)
    z_min = minimum(z_profile)

    anim = @animate for i in 1:length(t)
        @info "Animating $profile_type frame $i/$(length(t))"
        fig = plot(truth_profile[:,i], z_profile, xlim=(profile_min, profile_max), ylim=(z_min, z_max), label="Truth", title="$(round(t[i]/86400, digits=2)) days")
        plot!(fig, test_profile[:,i], z_profile, label="NN")
        ylabel!(fig, "z /m")

        if dimensionless
            xlabel!(fig, profile_type)
        elseif profile_type == "T"
            xlabel!(fig, "T /K")
        else
            xlabel!(fig, "$profile_type /m s⁻¹")
        end

        if i == 1
            savefig(fig, "$FILE_PATH.pdf")
            savefig(fig, "$FILE_PATH.png")
        end
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_flux(data, flux_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_flux = data["truth_$flux_type"]
    test_flux = data["test_$flux_type"]

    flux_max = maximum([maximum(truth_flux), maximum(test_flux)])
    flux_min = minimum([minimum(truth_flux), minimum(test_flux)])

    t = data["t"]

    z_flux = data["depth_flux"]

    z_max = maximum(z_flux)
    z_min = minimum(z_flux)

    anim = @animate for i in 1:length(t)
        @info "Animating $flux_type frame $i/$(length(t))"
        fig = plot(truth_flux[:,i], z_flux, xlim=(flux_min, flux_max), ylim=(z_min, z_max), label="Truth", title="$(round(t[i]/86400, digits=2)) days")
        plot!(fig, test_flux[:,i], z_flux, label = "NN")
        ylabel!(fig, "z /m")

        if dimensionless
            xlabel!(fig, flux_type)
        else
            xlabel!(fig, "$flux_type /m² s⁻²")
        end

    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profile_flux(data, profile_type, flux_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_flux = data["truth_$flux_type"]
    test_flux = data["test_$flux_type"]

    truth_profile = data["truth_$profile_type"]
    test_profile = data["test_$profile_type"]

    flux_max = maximum([maximum(truth_flux), maximum(test_flux)])
    flux_min = minimum([minimum(truth_flux), minimum(test_flux)])

    profile_max = maximum([maximum(truth_profile), maximum(test_profile)])
    profile_min = minimum([minimum(truth_profile), minimum(test_profile)])

    t = data["t"]

    z_flux = data["depth_flux"]
    z_profile = data["depth_profile"]

    z_max = maximum([maximum(z_flux), maximum(z_profile)])
    z_min = minimum([minimum(z_flux), minimum(z_profile)])

    anim = @animate for i in 1:length(t)
        @info "Animating $flux_type/$profile_type frame $i/$(length(t))"
        l = @layout [a b]
        fig₁ = plot(truth_flux[:,i], z_flux, xlim=(flux_min, flux_max), ylim=(z_min, z_max), label="Truth")
        plot!(fig₁, test_flux[:,i], z_flux, label = "NN")
        ylabel!(fig₁, "z /m")
        if dimensionless
            xlabel!(fig₁, flux_type)
        elseif flux_type == "wT"
            xlabel!(fig₁, "$flux_type /m s⁻¹ °C")
        else
            xlabel!(fig₁, "$flux_type /m² s⁻²")
        end

        fig₂ = plot(truth_profile[:,i], z_profile, xlim=(profile_min, profile_max), ylim=(z_min, z_max), label="Truth", legend=:topleft)
        plot!(fig₂, test_profile[:,i], z_profile, label="NN")
        ylabel!(fig₂, "z /m")
        if dimensionless
            xlabel!(fig₂, profile_type)
        elseif profile_type == "T"
            xlabel!(fig₂, "T /°C")
        else
            xlabel!(fig₂, "$profile_type /m s⁻¹")
        end

        fig = plot(fig₁, fig₂, layout=l, title="$(round(t[i]/86400, digits=2)) days")
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profiles(data, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_u = data["truth_u"]
    truth_v = data["truth_v"]
    truth_T = data["truth_T"]

    test_u = data["test_u"]
    test_v = data["test_v"]
    test_T = data["test_T"]

    u_max = maximum([maximum(truth_u), maximum(test_u)])
    u_min = minimum([minimum(truth_u), minimum(test_u)])

    v_max = maximum([maximum(truth_v), maximum(test_v)])
    v_min = minimum([minimum(truth_v), minimum(test_v)])
    
    T_max = maximum([maximum(truth_T), maximum(test_T)])
    T_min = minimum([minimum(truth_T), minimum(test_T)])

    t = data["t"]

    z = data["depth_profile"]

    z_max = maximum(z)
    z_min = minimum(z)

    anim = @animate for i in 1:length(t)
        if i % 50 == 0
            @info "Animating frame $i/$(length(t))"
        end
        l = @layout [a b c]
        fig₁ = Plots.plot(truth_u[:,i], z, xlim=(u_min, u_max), ylim=(z_min, z_max), label="Truth", legend=:bottomright)
        Plots.plot!(fig₁, test_u[:,i], z, label = "NN")
        Plots.ylabel!(fig₁, "z /m")
        if dimensionless
            Plots.xlabel!(fig₁, "u")
        else
            Plots.xlabel!(fig₁, "u /m s⁻¹")
        end

        fig₂ = Plots.plot(truth_v[:,i], z, xlim=(v_min, v_max), ylim=(z_min, z_max), label="Truth", legend=:bottomleft)
        Plots.plot!(fig₂, test_v[:,i], z, label = "NN")
        Plots.ylabel!(fig₂, "z /m")
        if dimensionless
            Plots.xlabel!(fig₂, "v")
        else
            Plots.xlabel!(fig₂, "v /m s⁻¹")
        end

        fig₃ = Plots.plot(truth_T[:,i], z, xlim=(T_min, T_max), ylim=(z_min, z_max), label="Truth", legend=:bottomright)
        Plots.plot!(fig₃, test_T[:,i], z, label = "NN")
        Plots.ylabel!(fig₃, "z /m")
        if dimensionless
            Plots.xlabel!(fig₃, "T")
        else
            Plots.xlabel!(fig₃, "T /°C")
        end

        fig = Plots.plot(fig₁, fig₂, fig₃, layout=l, title="$(round(t[i]/86400, digits=2)) days")
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profiles_fluxes(data, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true, SIMULATION_NAME="")
    times = data["t"]

    frame = Node(1)

    truth_u = @lift data["truth_u"][:,$frame]
    truth_v = @lift data["truth_v"][:,$frame]
    truth_T = @lift data["truth_T"][:,$frame]

    test_u = @lift data["test_u"][:,$frame]
    test_v = @lift data["test_v"][:,$frame]
    test_T = @lift data["test_T"][:,$frame]

    truth_uw = @lift data["truth_uw"][:,$frame]
    truth_vw = @lift data["truth_vw"][:,$frame]
    truth_wT = @lift data["truth_wT"][:,$frame]

    test_uw = @lift data["test_uw"][:,$frame]
    test_vw = @lift data["test_vw"][:,$frame]
    test_wT = @lift data["test_wT"][:,$frame]


    truth_Ri = @lift clamp.(data["truth_Ri"][:,$frame], -1, 2)
    test_Ri = @lift clamp.(data["test_Ri"][:,$frame], -1, 2)

    u_max = maximum([maximum(data["truth_u"]), maximum(data["test_u"])])
    u_min = minimum([minimum(data["truth_u"]), minimum(data["test_u"])])

    v_max = maximum([maximum(data["truth_v"]), maximum(data["test_v"])])
    v_min = minimum([minimum(data["truth_v"]), minimum(data["test_v"])])

    T_max = maximum([maximum(data["truth_T"]), maximum(data["test_T"])])
    T_min = minimum([minimum(data["truth_T"]), minimum(data["test_T"])])

    uw_max = maximum([maximum(data["truth_uw"]), maximum(data["test_uw"])])
    uw_min = minimum([minimum(data["truth_uw"]), minimum(data["test_uw"])])

    vw_max = maximum([maximum(data["truth_vw"]), maximum(data["test_vw"])])
    vw_min = minimum([minimum(data["truth_vw"]), minimum(data["test_vw"])])
    
    wT_max = maximum([maximum(data["truth_wT"]), maximum(data["test_wT"])])
    wT_min = minimum([minimum(data["truth_wT"]), minimum(data["test_wT"])])

    plot_title = @lift "$SIMULATION_NAME: time = $(round(times[$frame]/86400, digits=2)) days, loss = $(round(data["loss"], sigdigits=3))"
    fig = Figure(resolution=(1920, 1080))
    colors=["navyblue", "hotpink2"]

    if dimensionless
        u_str = "u"
        v_str = "v"
        T_str = "T"
        uw_str = "uw"
        vw_str = "vw"
        wT_str = "wT"
    else
        u_str = "u / m s⁻¹"
        v_str = "v / m s⁻¹"
        T_str = "T / °C"
        uw_str = "uw / m² s⁻²"
        vw_str = "vw / m² s⁻²"
        wT_str = "wT / m s⁻¹ °C"
    end

    zc = data["depth_profile"]
    zf = data["depth_flux"]
    z_str = "z / m"

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, truth_u, zc, linewidth=3, color=colors[1]), lines!(ax_u, test_u, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, truth_v, zc, linewidth=3, color=colors[1]), lines!(ax_v, test_v, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, truth_T, zc, linewidth=3, color=colors[1]), lines!(ax_T, test_T, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, truth_uw, zf, linewidth=3, color=colors[1]), lines!(ax_uw, test_uw, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, truth_vw, zf, linewidth=3, color=colors[1]), lines!(ax_vw, test_vw, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, truth_wT, zf, linewidth=3, color=colors[1]), lines!(ax_wT, test_wT, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[2, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, truth_Ri, zf, linewidth=3, color=colors[1]), lines!(ax_Ri, test_Ri, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NDE Prediction"])
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    trim!(fig.layout)

    if gif
        CairoMakie.record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            @info "Animating gif frame $n/$(length(times))..."
            frame[] = n
        end
    end

    if mp4
        CairoMakie.record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            @info "Animating mp4 frame $n/$(length(times))..."
            frame[] = n
        end
    end
end

function animate_profiles_fluxes_comparison(data_diffeq_explicit, data_diffeq_implicit, data_oceananigans, FILE_PATH; animation_type, n_trainings, training_types, fps=30, gif=false, mp4=true)
    times = data_diffeq_explicit["t"] ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

    u_data = [
        data_diffeq_explicit["truth_u"],
        data_diffeq_explicit["test_u_modified_pacanowski_philander"],
        data_diffeq_explicit["test_u_kpp"],
        data_oceananigans["test_u"],
        data_diffeq_explicit["test_u"], 
        data_diffeq_implicit["test_u"],
    ]

    v_data = [
        data_diffeq_explicit["truth_v"],
        data_diffeq_explicit["test_v_modified_pacanowski_philander"],
        data_diffeq_explicit["test_v_kpp"],
        data_oceananigans["test_v"],
        data_diffeq_explicit["test_v"], 
        data_diffeq_implicit["test_v"],
    ]

    T_data = [
        data_diffeq_explicit["truth_T"],
        data_diffeq_explicit["test_T_modified_pacanowski_philander"],
        data_diffeq_explicit["test_T_kpp"],
        data_oceananigans["test_T"],
        data_diffeq_explicit["test_T"], 
        data_diffeq_implicit["test_T"],
    ]
    
    uw_data = [
        data_diffeq_explicit["truth_uw"],
        data_diffeq_explicit["test_uw_modified_pacanowski_philander"],
        data_diffeq_explicit["test_uw_kpp"],
        data_oceananigans["test_uw"],
        data_diffeq_explicit["test_uw"], 
        data_diffeq_implicit["test_uw"],
        data_diffeq_explicit["test_uw_NN_only"][2:end-1,:],
    ]

    vw_data = [
        data_diffeq_explicit["truth_vw"],
        data_diffeq_explicit["test_vw_modified_pacanowski_philander"],
        data_diffeq_explicit["test_vw_kpp"],
        data_oceananigans["test_vw"],
        data_diffeq_explicit["test_vw"], 
        data_diffeq_implicit["test_vw"],
        data_diffeq_explicit["test_vw_NN_only"][2:end-1,:],
    ]

    wT_data = [
        data_diffeq_explicit["truth_wT"],
        data_diffeq_explicit["test_wT_modified_pacanowski_philander"],
        data_diffeq_explicit["test_wT_kpp"],
        data_oceananigans["test_wT"],
        data_diffeq_explicit["test_wT"], 
        data_diffeq_implicit["test_wT"],
        data_diffeq_explicit["test_wT_NN_only"][2:end-1,:],
    ]

    Ri_data = [
        clamp.(data_diffeq_explicit["truth_Ri"], -1, 2),
        clamp.(data_diffeq_explicit["test_Ri_modified_pacanowski_philander"], -1, 2),
        clamp.(data_diffeq_explicit["test_Ri_kpp"], -1, 2),
        clamp.(data_oceananigans["test_Ri"], -1, 2),
        clamp.(data_diffeq_explicit["test_Ri"], -1, 2),
        clamp.(data_diffeq_implicit["test_Ri"], -1, 2),
    ]

    @inline function lowclamp(value, lo)
        if value >= lo
            return value
        else
            return lo
        end
    end

    losses_data = [
        lowclamp.(data_diffeq_explicit["losses_modified_pacanowski_philander"], 1f-5),
        lowclamp.(data_diffeq_explicit["losses_kpp"], 1f-5),
        lowclamp.(data_oceananigans["losses"], 1f-5),
        lowclamp.(data_diffeq_explicit["losses"], 1f-5),
        lowclamp.(data_diffeq_implicit["losses"], 1f-5),
        lowclamp.(data_diffeq_explicit["losses_modified_pacanowski_philander_gradient"], 1f-5),
        lowclamp.(data_diffeq_explicit["losses_kpp_gradient"], 1f-5),
        lowclamp.(data_oceananigans["losses_gradient"], 1f-5),
        lowclamp.(data_diffeq_explicit["losses_gradient"], 1f-5),
        lowclamp.(data_diffeq_implicit["losses_gradient"], 1f-5),
    ]

    @inline function add_ϵ!(losses)
        losses .= losses .+ (losses .== 0) .* eps(Float32)
    end

    add_ϵ!.(losses_data)

    u_frames = [@lift data[:,$frame] for data in u_data]
    v_frames = [@lift data[:,$frame] for data in v_data]
    T_frames = [@lift data[:,$frame] for data in T_data]

    uw_frames = [@lift data[:,$frame] for data in uw_data]
    vw_frames = [@lift data[:,$frame] for data in vw_data]
    wT_frames = [@lift data[:,$frame] for data in wT_data]
    
    Ri_frames = [@lift data[:,$frame] for data in Ri_data]

    losses_point_frames = [@lift [data[$frame]] for data in losses_data]

    @inline function find_lims(datasets)
        return maximum(maximum.(datasets)), minimum(minimum.(datasets))
    end

    u_max, u_min = find_lims(u_data)
    v_max, v_min = find_lims(v_data)
    T_max, T_min = find_lims(T_data)

    uw_max, uw_min = find_lims(uw_data)
    vw_max, vw_min = find_lims(vw_data)
    wT_max, wT_min = find_lims(wT_data)
    
    losses_max, losses_min = find_lims(losses_data)

    train_parameters = data_diffeq_explicit["train_parameters"]
    ν₀ = train_parameters.ν₀
    ν₋ = train_parameters.ν₋
    ΔRi = train_parameters.ΔRi
    Riᶜ = train_parameters.Riᶜ
    Pr = train_parameters.Pr
    loss_scalings = train_parameters.loss_scalings

    BC_str = @sprintf "Momentum Flux = %.1e m² s⁻², Buoyancy Flux = %.1e m² s⁻³" data_diffeq_explicit["truth_uw"][end, 1] data_diffeq_explicit["truth_wT"][end, 1]
    plot_title = @lift "$animation_type Data: $BC_str, Time = $(round(times[$frame], digits=2)) days"

    diffusivity_str = @sprintf "ν₀ = %.1e m² s⁻¹, ν₋ = %.1e m² s⁻¹, ΔRi = %.1e, Riᶜ = %.2f, Pr=%.1f" ν₀ ν₋ ΔRi Riᶜ Pr 

    scaling_str = @sprintf "Loss Scalings: u = %.1e, v = %.1e, T = %.1e, ∂u∂z = %.1e, ∂v∂z = %.1e, ∂T∂z = %.1e" loss_scalings.u loss_scalings.v loss_scalings.T loss_scalings.∂u∂z loss_scalings.∂v∂z loss_scalings.∂T∂z

    plot_subtitle = "$n_trainings Training Simulations ($training_types): $diffusivity_str \n $scaling_str"

    fig = Figure(resolution=(1920, 1080))
    
    colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    u_str = "u / m s⁻¹"
    v_str = "v / m s⁻¹"
    T_str = "T / °C"
    uw_str = "uw / m² s⁻²"
    vw_str = "vw / m² s⁻²"
    wT_str = "wT / m s⁻¹ °C"

    zc = data_diffeq_explicit["depth_profile"]
    zf = data_diffeq_explicit["depth_flux"]
    zf_interior = zf[2:end-1]
    z_str = "z / m"

    alpha=0.5
    truth_linewidth = 7
    linewidth = 3


    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    ax_T = fig[1, 3:4] = Axis(fig, xlabel=T_str, ylabel=z_str)
    ax_Ri = fig[1, 5] = Axis(fig, xlabel="Ri", ylabel=z_str)
    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    ax_losses = fig[2, 4] = Axis(fig, xlabel="Time / days", ylabel="Loss", yscale=CairoMakie.log10)
    
    legend_sublayout = GridLayout()
    fig[2, 5] = legend_sublayout

    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.xlims!(ax_losses, times[1], times[end])

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)
    CairoMakie.ylims!(ax_losses, losses_min, losses_max)

    u_lines = [
         lines!(ax_u, u_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_u, u_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
    ]

    v_lines = [
         lines!(ax_v, v_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_v, v_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
    ]

    T_lines = [
         lines!(ax_T, T_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_T, T_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(T_data)]
    ]

    uw_lines = [
         lines!(ax_uw, uw_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
         lines!(ax_uw, uw_frames[end], zf_interior, linewidth=3, color=colors[end-1]);
        [lines!(ax_uw, uw_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(uw_data)-1]
    ]

   vw_lines = [
         lines!(ax_vw, vw_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
         lines!(ax_vw, vw_frames[end], zf_interior, linewidth=3, color=colors[end-1]);
        [lines!(ax_vw, vw_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(vw_data)-1]
    ]

    wT_lines = [
        lines!(ax_wT, wT_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        lines!(ax_wT, wT_frames[end], zf_interior, linewidth=3, color=colors[end-1]);
       [lines!(ax_wT, wT_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(wT_data)-1]
   ]

    Ri_lines = [
         lines!(ax_Ri, Ri_frames[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_Ri, Ri_frames[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(Ri_data)]
    ]

    losses_lines =  [
        [lines!(ax_losses, times, losses_data[i], linewidth=linewidth, color=colors[i+1]) for i in 1:Int(length(losses_data)/2)];
        [lines!(ax_losses, times, losses_data[Int(i+length(losses_data)/2)], linewidth=linewidth, color=colors[i+1], linestyle=:dot) for i in 1:Int(length(losses_data)/2)];
    ]

    losses_point = [CairoMakie.scatter!(ax_losses, time_point, point, color=colors[end]) for point in losses_point_frames] 
    
    legend = Legend(fig, uw_lines, ["LES", 
                                    "NN Only",
                                    "Modified Pac-Phil Only", 
                                    "KPP",
                                    "Oceananigans, Implicit",
                                    "DiffEq, Explicit",
                                    "DiffEq, Implicit"
                                    ])

    legend_loss = Legend(fig, losses_lines, ["Profile Loss, Modified Pac-Phil Only", 
                                             "Profile Loss, KPP",
                                             "Profile Loss, Oceananigans, Implicit", 
                                             "Profile Loss, DiffEq, Explicit", 
                                             "Profile Loss, DiffEq, Implicit",
                                             "Gradient Loss, Modified Pac-Phil Only", 
                                             "Gradient Loss, KPP",
                                             "Gradient Loss, Oceananigans, Implicit", 
                                             "Gradient Loss, DiffEq, Explicit", 
                                             "Gradient Loss, DiffEq, Implicit"])
    legend_sublayout[:v] = [legend, legend_loss]

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=25)
    subtitle = fig[end+1, :] = Label(fig, text=plot_subtitle, textsize=20)
    
    trim!(fig.layout)

    print_frame = maximum([1, Int(floor(length(times)/20))])

    function print_progress(n, n_total, print_frame, type)
        if n % print_frame == 0
            @info "Animating $(type) frame $n/$n_total"
        end
    end

    @info "Starting Animation"

    if gif
        CairoMakie.record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            print_progress(n, length(times), print_frame, "gif")
            frame[] = n
        end
    end

    if mp4
        CairoMakie.record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            print_progress(n, length(times), print_frame, "mp4")
            frame[] = n
        end
    end
end

function animate_training_data_profiles_fluxes(train_files, FILE_PATH; fps=30, gif=false, mp4=true)
    all_data = [WindMixing.data(train_file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false) for train_file in train_files]

    times = all_data[1].t ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

    u_data = [data.u.coarse for data in all_data]
    v_data = [data.v.coarse for data in all_data]
    T_data = [data.T.coarse for data in all_data]

    us = [@lift u[:,$frame] for u in u_data]
    vs = [@lift v[:,$frame] for v in v_data]
    Ts = [@lift T[:,$frame] for T in T_data]

    uws = [@lift data.uw.coarse[:,$frame] for data in all_data]
    vws = [@lift data.vw.coarse[:,$frame] for data in all_data]
    wTs = [@lift data.wT.coarse[:,$frame] for data in all_data]

    u_max = maximum(maximum(data.u.coarse) for data in all_data)
    u_min = minimum(minimum(data.u.coarse) for data in all_data)

    v_max = maximum(maximum(data.v.coarse) for data in all_data)
    v_min = minimum(minimum(data.v.coarse) for data in all_data)

    T_max = maximum(maximum(data.T.coarse) for data in all_data)
    T_min = minimum(minimum(data.T.coarse) for data in all_data)

    uw_max = maximum(maximum(data.uw.coarse) for data in all_data)
    uw_min = minimum(minimum(data.uw.coarse) for data in all_data)

    vw_max = maximum(maximum(data.vw.coarse) for data in all_data)
    vw_min = minimum(minimum(data.vw.coarse) for data in all_data)
    
    wT_max = maximum(maximum(data.wT.coarse) for data in all_data)
    wT_min = minimum(minimum(data.wT.coarse) for data in all_data)

    Nz = all_data[1].grid_points - 1
    zc = all_data[1].u.z
    zf = all_data[1].uw.z

    D_face = Float32.(Dᶠ(Nz, zc[2] - zc[1]))

    @inline function ∂_∂z(profile)
        output = zeros(typeof(profile[1]), size(profile, 1) + 1, size(profile,2))
        for i in 1:size(profile,2)
            profile_col = @view profile[:,i]
            output_col = @view output[:,i]
            output_col .= D_face * profile_col
        end
        return output
    end

    ∂u∂zs = [∂_∂z(u) for u in u_data]
    ∂v∂zs = [∂_∂z(v) for v in v_data]
    ∂T∂zs = [∂_∂z(T) for T in T_data]

    @inline function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, g, α)
        # ϵ = eps(typeof(∂u∂z))
        ϵ = 0
        ∂u∂z += ϵ
        ∂v∂z += ϵ
        ∂T∂z += ϵ
        Bz = g * α * ∂T∂z
        S² = ∂u∂z ^2 + ∂v∂z ^2
        return clamp.(Bz / S², -1, 2)
    end

    α = 1.67f-4
    g = 9.81f0

    Ris_data = [local_richardson.(∂u∂zs[i], ∂v∂zs[i], ∂T∂zs[i], g, α) for i in 1:length(∂u∂zs)]

    Ris = [@lift Ri[:,$frame] for Ri in Ris_data]

    plot_title = @lift "LES Simulations: Time = $(round(times[$frame], digits=2)) days"
    fig = Figure(resolution=(1920, 1080))
    color_palette = distinguishable_colors(length(all_data), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    u_str = "u / m s⁻¹"
    v_str = "v / m s⁻¹"
    T_str = "T / °C"
    uw_str = "uw / m² s⁻²"
    vw_str = "vw / m² s⁻²"
    wT_str = "wT / m s⁻¹ °C"

    z_str = "z / m"

    uw_tops = [data.uw.coarse[end,1] for data in all_data]
    wT_tops = [data.wT.coarse[end,1] for data in all_data]

    BC_strs = [@sprintf "Momentum Flux = %.1e m² s⁻², Buoyancy Flux = %.1e m² s⁻³" uw_tops[i] wT_tops[i] for i in 1:length(uw_tops)]

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, us[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(us)]
                # lines!(ax_u, test_u_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, vs[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(vs)]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, Ts[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(Ts)]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, uws[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(uws)]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, vws[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(vws)]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, wTs[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(wTs)]
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[2, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, Ris[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(Ris)]
                # lines!(ax_Ri, test_Ri_NN_only, zf, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    legend = fig[1, 4] = Legend(fig, uw_lines, BC_strs)

    # legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NN + Modified Pac-Phil", "Modified Pac-Phil Only"])
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    trim!(fig.layout)

    if gif
        CairoMakie.record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            @info "Animating gif frame $n/$(length(times))..."
            frame[] = n
        end
    end

    if mp4
        CairoMakie.record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            @info "Animating mp4 frame $n/$(length(times))..."
            frame[] = n
        end
    end
end


function generate_training_types_str(FILE_NAME)
    training_types = ""
    check_exists(str) = occursin(str, FILE_NAME)

    if check_exists("_wind_mixing_")
        training_types *= "Wind Mixing"
    end

    if check_exists("_cooling_")
        if training_types != ""
            training_types *= ", Cooling"
        else
            training_types *= "Cooling"
        end
    end

    if check_exists("_heating_")
        if training_types != ""
            training_types *= ", Heating"
        else
            training_types *= "Heating"
        end
    end

    if check_exists("_windcooling_")
        if training_types != ""
            training_types *= ", Wind + Cooling"
        else
            training_types *= "Wind + Cooling"
        end
    end

    if check_exists("_windheating_")
        if training_types != ""
            training_types *= ", Wind + Heating"
        else
            training_types *= "Wind + Heating"
        end
    end

    return training_types
end

function animate_training_results(test_files, FILE_NAME; 
                                  EXTRACTED_DATA_DIR, OUTPUT_DIR,
                                  timestep=60, trange=1:1:1153, 
                                  fps=30, gif=false, mp4=true,
                                  explicit_timestepper=ROCK4(), implicit_timestepper=RadauIIA5(autodiff=false),
                                  convective_adjustment=false)

    DATA_PATH = joinpath(EXTRACTED_DATA_DIR, "$(FILE_NAME)_extracted.jld2")
    OUTPUT_PATH = joinpath(OUTPUT_DIR, FILE_NAME)

    if !ispath(OUTPUT_PATH)
        mkdir(OUTPUT_PATH)
    end
    
    @info "Loading Data"
    file = jldopen(DATA_PATH, "r")
    if haskey(file, "losses/total")
        losses = (
            u = file["losses/u"],
            v = file["losses/v"],
            T = file["losses/T"],
            ∂u∂z = file["losses/∂u∂z"],
            ∂v∂z = file["losses/∂v∂z"],
            ∂T∂z = file["losses/∂T∂z"],
        )
        @info "Training Loss = $(minimum(file["losses/total"]))"
    else
        losses = file["losses"]
        @info "Training Loss = $(minimum(losses))"
    end

    train_files = file["training_info/train_files"]

    diurnal = occursin("diurnal", train_files[1])

    train_parameters = file["training_info/parameters"]
    uw_NN = file["neural_network/uw"]
    vw_NN = file["neural_network/vw"]
    wT_NN = file["neural_network/wT"]

    if haskey(file["training_info"], "loss_scalings")
        loss_scalings = file["training_info/loss_scalings"]
    elseif haskey(train_parameters, "gradient_scaling")
        gradient_scaling = train_parameters["gradient_scaling"]
        loss_scalings = (u=1f0, v=1f0, T=1f0, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
    else
        loss_scalings = (u=1f0, v=1f0, T=1f0, ∂u∂z=1f0, ∂v∂z=1f0, ∂T∂z=1f0)
    end

    close(file)

    @info "Loading Training Data"
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    training_types = generate_training_types_str(FILE_NAME)

    ν₀ = train_parameters["ν₀"]
    ν₋ = train_parameters["ν₋"]
    ΔRi = train_parameters["ΔRi"]
    Riᶜ = train_parameters["Riᶜ"]
    Pr = train_parameters["Pr"]

    @info "Plotting Loss..."
    plot_loss(losses, joinpath(OUTPUT_PATH, "loss.pdf"))

    @info "Solving NDE: Oceananigans"
    solve_oceananigans_modified_pacanowski_philander_nn(test_files, DATA_PATH, OUTPUT_PATH; timestep=timestep, convective_adjustment=convective_adjustment)

    for test_file in test_files
        @info "Generating Data: $test_file"
        𝒟test = WindMixing.data(test_file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

        @info "Processing $test_file solution"
        if test_file in train_files
            SOL_DIR = joinpath(OUTPUT_PATH, "train_$test_file")
            animation_type = "Training"
        else
            SOL_DIR = joinpath(OUTPUT_PATH, "test_$test_file")
            animation_type = "Testing"
        end

        @info "Solving NDE: $test_file, Explicit timestepper"
        plot_data_explicit = NDE_profile(uw_NN, vw_NN, wT_NN, test_file, 𝒟test, 𝒟train, trange,
                                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                                ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr,
                                convective_adjustment=false,
                                smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"],
                                zero_weights=train_parameters["zero_weights"],
                                loss_scalings=loss_scalings,
                                timestepper=explicit_timestepper,
                                OUTPUT_PATH=joinpath(SOL_DIR, "solution_diffeq_explicit.jld2"))

        @info "Solving NDE: $test_file, Implicit timestepper"
        plot_data_implicit = NDE_profile(uw_NN, vw_NN, wT_NN, test_file, 𝒟test, 𝒟train, trange,
                                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                                ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr,
                                convective_adjustment=false,
                                smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"],
                                zero_weights=train_parameters["zero_weights"],
                                loss_scalings=loss_scalings,
                                timestepper=implicit_timestepper,
                                OUTPUT_PATH=joinpath(SOL_DIR, "solution_diffeq_implicit.jld2"))

        @info "Solving NDE: $test_file, Oceananigans"

        plot_data_oceananigans = NDE_profile_oceananigans(SOL_DIR, train_files, [test_file],
                                            ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr,
                                            loss_scalings=loss_scalings,
                                            OUTPUT_PATH=joinpath(SOL_DIR, "solution_oceananigans.jld2"))
        
        if test_file in train_files
            animation_type = "Training"
        else
            animation_type = "Testing"
        end
        n_trainings = length(train_files)

        if animation_type == "Training"
            VIDEO_NAME = "train_$test_file"
        else
            VIDEO_NAME = "test_$test_file"
        end

        VIDEO_PATH = joinpath(SOL_DIR, "$VIDEO_NAME")

        @info "Animating $test_file Video"

        animate_profiles_fluxes_comparison(plot_data_explicit, plot_data_implicit, plot_data_oceananigans, VIDEO_PATH, fps=fps, gif=gif, mp4=mp4, 
                                                animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
        @info "$test_file Animation Completed"
    end
end


function animate_profiles_fluxes_final(data, axis_images, FILE_PATH; animation_type, n_trainings, training_types, fps=30, gif=false, mp4=true)
    times = data["t"] ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

    u_data = [
        data["truth_u"],
        data["test_u_modified_pacanowski_philander"],
        data["test_u_kpp"],
        data["test_u"],
    ]

    v_data = [
        data["truth_v"],
        data["test_v_modified_pacanowski_philander"],
        data["test_v_kpp"],
        data["test_v"],
    ]

    T_data = [
        data["truth_T"],
        data["test_T_modified_pacanowski_philander"],
        data["test_T_kpp"],
        data["test_T"],
    ]
    
    uw_data = [
        data["truth_uw"],
        data["test_uw_modified_pacanowski_philander"],
        data["test_uw_kpp"],
        data["test_uw"],
    ]

    vw_data = [
        data["truth_vw"],
        data["test_vw_modified_pacanowski_philander"],
        data["test_vw_kpp"],
        data["test_vw"],
    ]

    wT_data = [
        data["truth_wT"],
        data["test_wT_modified_pacanowski_philander"],
        data["test_wT_kpp"],
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

    # @inline function lowclamp(value, lo)
    #     if value >= lo
    #         return value
    #     else
    #         return lo
    #     end
    # end

    # losses_data = [
    #     lowclamp.(data["losses_modified_pacanowski_philander"] .+ data["losses_modified_pacanowski_philander_gradient"], 1f-5),
    #     lowclamp.(data["losses_kpp"] .+ data["losses_kpp_gradient"], 1f-5),
    #     lowclamp.(data["losses"] .+ data["losses_gradient"], 1f-5),
    # ]

    u_frames = [@lift data[:,$frame] for data in u_data]
    v_frames = [@lift data[:,$frame] for data in v_data]
    T_frames = [@lift data[:,$frame] for data in T_data]

    uw_frames = [@lift data[:,$frame] for data in uw_data]
    vw_frames = [@lift data[:,$frame] for data in vw_data]
    wT_frames = [@lift data[:,$frame] for data in wT_data]
    
    Ri_frames = [@lift data[:,$frame] for data in Ri_data]

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
    
    # losses_max, losses_min = find_lims(losses_data)

    train_parameters = data["train_parameters"]
    ν₀ = train_parameters.ν₀
    ν₋ = train_parameters.ν₋
    ΔRi = train_parameters.ΔRi
    Riᶜ = train_parameters.Riᶜ
    Pr = train_parameters.Pr
    loss_scalings = train_parameters.loss_scalings

    BC_str = @sprintf "Momentum Flux = %.1e m² s⁻², Temperature Flux = %.1e m s⁻¹ °C" data["truth_uw"][end, 1] maximum(data["truth_wT"][end, :])
    plot_title = @lift "$animation_type Data: $BC_str, Time = $(round(times[$frame], digits=2)) days"

    diffusivity_str = @sprintf "ν₀ = %.2e m² s⁻¹, ν₋ = %.2e m² s⁻¹, ΔRi = %.2e, Riᶜ = %.3f, Pr=%.2f" ν₀ ν₋ ΔRi Riᶜ Pr 

    # scaling_str = @sprintf "Loss Scalings: u = %.1e, v = %.1e, T = %.1e, ∂u∂z = %.1e, ∂v∂z = %.1e, ∂T∂z = %.1e" loss_scalings.u loss_scalings.v loss_scalings.T loss_scalings.∂u∂z loss_scalings.∂v∂z loss_scalings.∂T∂z
    # plot_subtitle = "$n_trainings Training Simulations ($training_types): $diffusivity_str \n $scaling_str"

    plot_subtitle = "$n_trainings Training Simulations ($training_types): $diffusivity_str"

    # fig = Figure(resolution=(1920, 1080))
    fig = Figure(resolution=(1920, 960))
    
    # colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    colors = distinguishable_colors(length(uw_data), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    temp_color = colors[2]
    colors[2] = colors[4]
    colors[4] = temp_color
    # colors[4] = RGB(0,0,0)

    u_img = axis_images.u
    v_img = axis_images.v
    T_img = axis_images.T
    uw_img = axis_images.uw
    vw_img = axis_images.uw
    wT_img = axis_images.uw
    Ri_img = axis_images.uw
    z_img = axis_images.z

    zc = data["depth_profile"]
    zf = data["depth_flux"]
    zf_interior = zf[2:end-1]

    rel_size = 30
    # aspect = 1920 / 1080
    aspect = 2


    ax_u = fig[1, 2] = Axis(fig)
    ax_v = fig[1, 4] = Axis(fig)

    T_layout = fig[1:4, 5] = GridLayout()
    colsize!(fig.layout, 5, CairoMakie.Relative(0.4))

    ax_T = T_layout[1, 2] = Axis(fig)
    y_ax_T = T_layout[1,1] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T = T_layout[2,2] = CairoMakie.Axis(fig, aspect=DataAspect())

    ax_Ri = fig[1, 7] = Axis(fig)
    ax_uw = fig[3, 2] = Axis(fig)
    ax_vw = fig[3, 4] = Axis(fig)
    ax_wT = fig[3, 7] = Axis(fig)
    
    y_ax_u = CairoMakie.Axis(fig[1,1], aspect=DataAspect())
    y_ax_v = CairoMakie.Axis(fig[1,3], aspect=DataAspect())
    y_ax_Ri = CairoMakie.Axis(fig[1,6], aspect=DataAspect())
    y_ax_uw = CairoMakie.Axis(fig[3,1], aspect=DataAspect())
    y_ax_vw = CairoMakie.Axis(fig[3,3], aspect=DataAspect())
    y_ax_wT = CairoMakie.Axis(fig[3,6], aspect=DataAspect())

    x_ax_u = CairoMakie.Axis(fig[2,2], aspect=DataAspect())
    x_ax_v = CairoMakie.Axis(fig[2,4], aspect=DataAspect())
    x_ax_Ri = CairoMakie.Axis(fig[2,7], aspect=DataAspect())
    x_ax_uw = CairoMakie.Axis(fig[4,2], aspect=DataAspect())
    x_ax_vw = CairoMakie.Axis(fig[4,4], aspect=DataAspect())
    x_ax_wT = CairoMakie.Axis(fig[4,7], aspect=DataAspect())

    hidedecorations!(y_ax_u)
    hidedecorations!(y_ax_v)
    hidedecorations!(y_ax_Ri)
    hidedecorations!(y_ax_uw)
    hidedecorations!(y_ax_vw)
    hidedecorations!(y_ax_wT)

    hidedecorations!(x_ax_u)
    hidedecorations!(x_ax_v)
    hidedecorations!(x_ax_Ri)
    hidedecorations!(x_ax_uw)
    hidedecorations!(x_ax_vw)
    hidedecorations!(x_ax_wT)

    hidespines!(y_ax_u)
    hidespines!(y_ax_v)
    hidespines!(y_ax_T)
    hidespines!(y_ax_Ri)
    hidespines!(y_ax_uw)
    hidespines!(y_ax_vw)
    hidespines!(y_ax_wT)

    hidespines!(x_ax_u)
    hidespines!(x_ax_v)
    hidespines!(x_ax_T)
    hidespines!(x_ax_Ri)
    hidespines!(x_ax_uw)
    hidespines!(x_ax_vw)
    hidespines!(x_ax_wT)
    
    image!(x_ax_u, axis_images.u)
    image!(x_ax_v, axis_images.v)
    image!(x_ax_Ri, axis_images.Ri)
    image!(x_ax_uw, axis_images.uw)
    image!(x_ax_vw, axis_images.vw)
    image!(x_ax_wT, axis_images.wT)

    image!(y_ax_u, axis_images.z)
    image!(y_ax_v, axis_images.z)
    image!(y_ax_Ri, axis_images.z)
    image!(y_ax_uw, axis_images.z)
    image!(y_ax_vw, axis_images.z)
    image!(y_ax_wT, axis_images.z)

    hidedecorations!(y_ax_T)
    hidedecorations!(x_ax_T)
    image!(x_ax_T, axis_images.T)
    image!(y_ax_T, axis_images.z)

    colsize!(T_layout, 1, CairoMakie.Relative(1 / rel_size * 1.5))
    rowsize!(T_layout, 2, CairoMakie.Relative(1 / rel_size / aspect * 1.75))
    colgap!(T_layout, Relative(1 / rel_size / aspect / 2))
    rowgap!(T_layout, Relative(1 / rel_size / aspect))

    rowsize!(fig.layout, 2, CairoMakie.Relative(1 / rel_size))
    rowsize!(fig.layout, 4, CairoMakie.Relative(1 / rel_size))
    colsize!(fig.layout, 1, CairoMakie.Relative(1 / rel_size / aspect))
    colsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size / aspect))
    colsize!(fig.layout, 6, CairoMakie.Relative(1 / rel_size / aspect))

    colgap!(fig.layout, Relative(1 / rel_size / aspect / 2))
    rowgap!(fig.layout, Relative(1 / rel_size / aspect))

    alpha=0.5
    truth_linewidth =8
    linewidth = 3
    
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.xlims!(ax_Ri, -1, 2)
    # CairoMakie.xlims!(ax_losses, times[1], times[end])

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)
    # CairoMakie.ylims!(ax_losses, losses_min, losses_max)

    u_lines = [
         lines!(ax_u, u_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_u, u_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
    ]

    v_lines = [
         lines!(ax_v, v_frames[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_v, v_frames[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
    ]

    T_lines = [
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

    axislegend(ax_T, T_lines, ["Oceananigans.jl Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], "Data Type", position = :rb)
    
    # legend = fig[3, 8] = Legend(fig, u_lines, ["LES", 
    #                                     "Ri-based Diffusivity Only", 
    #                                     "KPP",
    #                                     "Oceananigans.jl",
    #                                     ])

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=25)
    subtitle = fig[end+1, :] = Label(fig, text=plot_subtitle, textsize=20)
    
    trim!(fig.layout)

    print_frame = maximum([1, Int(floor(length(times)/20))])

    function print_progress(n, n_total, print_frame, type)
        if n % print_frame == 0
            @info "Animating $(type) frame $n/$n_total"
        end
    end

    @info "Starting Animation"

    if gif
        CairoMakie.record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            print_progress(n, length(times), print_frame, "gif")
            frame[] = n
        end
    end

    if mp4
        CairoMakie.record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps, compression=1) do n
            print_progress(n, length(times), print_frame, "mp4")
            frame[] = n
        end
    end

end

function animate_LES_3D(FILE_DIR, OUTPUT_PATH, axis_images; num_frames, simulation_str, fps=30, gif=false, mp4=true, colorscheme=:ice, rev=false)
    xy_file = jldopen(joinpath(FILE_DIR, "xy_slice.jld2"))
    xz_file = jldopen(joinpath(FILE_DIR, "xz_slice.jld2"))
    yz_file = jldopen(joinpath(FILE_DIR, "yz_slice.jld2"))
    instantaneous_statistics = jldopen(joinpath(FILE_DIR, "instantaneous_statistics.jld2"))

    iterations = keys(instantaneous_statistics["timeseries/t"])
    times = [instantaneous_statistics["timeseries/t/$iter"] for iter in iterations]
    xC = xz_file["grid/xC"][4:end-3]
    yC = xz_file["grid/yC"][4:end-3]
    zC = xz_file["grid/zC"][4:end-3]

    x_xz = fill(xC[1], 128)
    y_xz = yC

    z_xz = zeros(length(x_xz), length(y_xz))
    for i in 1:size(z_xz, 1)
        z_xz[i,:] .= zC[i]
    end

    T_xzs = [transpose(hcat([xz_file["timeseries/T/$iter"][:, :, i] for i in 1:length(zC)]... )) for iter in iterations]

    x_yz = xC
    y_yz = fill(yC[1], 128)

    z_yz = zeros(length(x_yz), length(y_yz))
    for i in 1:size(z_yz, 2)
        z_yz[:,i] .= zC[i]
    end

    @inline function obtain_T_yz(iter)
        T_yz = similar(z_yz)
        for i in 1:size(T_yz, 2)
            T_yz[:,i] = yz_file["timeseries/T/$iter"][:, :, i]
        end
        return T_yz
    end
    T_yzs = [obtain_T_yz(iter) for iter in iterations]

    x_xy = xC
    y_xy = yC
    z_xy = zeros(length(y_xy), length(x_xy))

    T_xys = [xy_file["timeseries/T/$iter"][:,:,1] for iter in iterations]

    us = [instantaneous_statistics["timeseries/u/$iter"][:] for iter in iterations]
    vs = [instantaneous_statistics["timeseries/v/$iter"][:] for iter in iterations]
    Ts = [instantaneous_statistics["timeseries/T/$iter"][:] for iter in iterations]

    @inline function find_lims(profiles)
        return minimum(minimum.(profiles)), maximum(maximum.(profiles))
    end

    u_min, u_max = find_lims(us)
    v_min, v_max = find_lims(vs)
    T_min, T_max = find_lims(Ts)

    color_range_max = maximum([maximum(maximum.(T_xzs)), maximum(maximum.(T_yzs)), maximum(maximum.(T_xys))])
    color_range_min = minimum([minimum(minimum.(T_xzs)), minimum(minimum.(T_yzs)), minimum(minimum.(T_xys))])

    color_range = (color_range_min, color_range_max)
    colormap = cgrad(colorscheme, scale=:log10, rev=rev)

    close(xy_file)
    close(xz_file)
    close(yz_file)
    close(instantaneous_statistics)

    frame = Node(1)
    iteration = @lift iterations[$frame]

    T_xz = @lift T_xzs[$frame]
    T_yz = @lift T_yzs[$frame]
    T_xy = @lift T_xys[$frame]

    u = @lift us[$frame]
    v = @lift vs[$frame]
    T = @lift Ts[$frame]

    fig = Figure(resolution=(1920, 1080))
    ax = fig[1,1] = CairoMakie.Axis3(fig, aspect=(1, 1, 0.5), xlabel="x /m", ylabel="y /m", zlabel="z /m")

    xz_surface = CairoMakie.surface!(ax, x_xz, y_xz, z_xz, color=T_xz, colormap=colormap, colorrange=color_range)

    yz_surface = CairoMakie.surface!(ax, x_yz, y_yz, z_yz, color=T_yz, colormap=colormap, colorrange=color_range)

    xy_surface = CairoMakie.surface!(ax, x_xy, y_xy, z_xy, color=T_xy, colormap=colormap, colorrange=color_range)

    ax_T_3D = fig[2,1] = CairoMakie.Axis(fig, aspect=DataAspect())

    rel_size = 40
    aspect = 1 / 4

    hidedecorations!(ax_T_3D)
    hidespines!(ax_T_3D)
    image!(ax_T_3D, axis_images.T_3D)
    rowsize!(fig.layout, 2, CairoMakie.Relative(1 / rel_size))

    colorbar = CairoMakie.Colorbar(fig[3, 1], xz_surface, vertical=false)

    plots_sublayout = fig[:,2] = GridLayout()

    colsize!(fig.layout, 2, CairoMakie.Relative(aspect))
    rowgap!(fig.layout, Relative(1 / 50 / 3))


    y_ax_u = plots_sublayout[1,1] = CairoMakie.Axis(fig, aspect=DataAspect())
    y_ax_v = plots_sublayout[3,1] = CairoMakie.Axis(fig, aspect=DataAspect())
    y_ax_T = plots_sublayout[5,1] = CairoMakie.Axis(fig, aspect=DataAspect())

    x_ax_u = plots_sublayout[2,2] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_v = plots_sublayout[4,2] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T = plots_sublayout[6,2] = CairoMakie.Axis(fig, aspect=DataAspect())

    ax_u = plots_sublayout[1,2] = CairoMakie.Axis(fig)
    ax_v = plots_sublayout[3,2] = CairoMakie.Axis(fig)
    ax_T = plots_sublayout[5,2] = CairoMakie.Axis(fig)

    hidedecorations!(y_ax_u)
    hidedecorations!(y_ax_v)
    hidedecorations!(y_ax_T)

    hidedecorations!(x_ax_u)
    hidedecorations!(x_ax_v)
    hidedecorations!(x_ax_T)

    hidespines!(y_ax_u)
    hidespines!(y_ax_v)
    hidespines!(y_ax_T)

    hidespines!(x_ax_u)
    hidespines!(x_ax_v)
    hidespines!(x_ax_T)

    image!(x_ax_u, axis_images.u)
    image!(x_ax_v, axis_images.v)
    image!(x_ax_T, axis_images.T)

    image!(y_ax_u, axis_images.z)
    image!(y_ax_v, axis_images.z)
    image!(y_ax_T, axis_images.z)

    rowsize!(plots_sublayout, 2, CairoMakie.Relative(1 / rel_size))
    rowsize!(plots_sublayout, 4, CairoMakie.Relative(1 / rel_size))
    rowsize!(plots_sublayout, 6, CairoMakie.Relative(1 / rel_size))
    colsize!(plots_sublayout, 1, CairoMakie.Aspect(2, 1))

    colgap!(plots_sublayout, 1 / 50)
    rowgap!(plots_sublayout, Relative(1 / 50 / 3))

    u_line = CairoMakie.lines!(ax_u, u, zC)
    v_line = CairoMakie.lines!(ax_v, v, zC)
    T_line = CairoMakie.lines!(ax_T, T, zC)

    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.xlims!(ax_T, T_min, T_max)

    CairoMakie.ylims!(ax_u, minimum(zC), 0)
    CairoMakie.ylims!(ax_v, minimum(zC), 0)
    CairoMakie.ylims!(ax_T, minimum(zC), 0)

    plot_title = @lift "$(simulation_str), Time = $(round(times[$frame]/86400, digits=2)) days"

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=25)

    trim!(fig.layout)

    function print_progress(n, n_total, type)
        @info "Animating $(type) frame $n/$n_total"
    end

    @info "Starting Animation"

    if num_frames == 0
        if gif
            CairoMakie.record(fig, "$OUTPUT_PATH.gif", 1:length(times), framerate=fps, compression=1) do n
                print_progress(n, length(times), "gif")
                frame[] = n
            end
        end

        if mp4
            CairoMakie.record(fig, "$OUTPUT_PATH.mp4", 1:length(times), framerate=fps, compression=1) do n
                print_progress(n, length(times), "mp4")
                frame[] = n
            end
        end
    else
        if gif
            CairoMakie.record(fig, "$OUTPUT_PATH.gif", 1:num_frames, framerate=fps, compression=1) do n
                print_progress(n, num_frames, "gif")
                frame[] = n
            end
        end

        if mp4
            CairoMakie.record(fig, "$OUTPUT_PATH.mp4", 1:num_frames, framerate=fps, compression=1) do n
                print_progress(n, num_frames, "mp4")
                frame[] = n
            end
        end
    end
end