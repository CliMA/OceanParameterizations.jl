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

function animate_profiles_fluxes_comparison(data, FILE_PATH; animation_type, n_trainings, training_types, fps=30, gif=false, mp4=true)
    times = data["t"] ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

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

    test_u_modified_pacanowski_philander = @lift data["test_u_modified_pacanowski_philander"][:,$frame]
    test_v_modified_pacanowski_philander = @lift data["test_v_modified_pacanowski_philander"][:,$frame]
    test_T_modified_pacanowski_philander = @lift data["test_T_modified_pacanowski_philander"][:,$frame]

    test_uw_modified_pacanowski_philander = @lift data["test_uw_modified_pacanowski_philander"][:,$frame]
    test_vw_modified_pacanowski_philander = @lift data["test_vw_modified_pacanowski_philander"][:,$frame]
    test_wT_modified_pacanowski_philander = @lift data["test_wT_modified_pacanowski_philander"][:,$frame]

    test_uw_NN_only = @lift data["test_uw_NN_only"][2:end-1,$frame]
    test_vw_NN_only = @lift data["test_vw_NN_only"][2:end-1,$frame]
    test_wT_NN_only = @lift data["test_wT_NN_only"][2:end-1,$frame]

    truth_Ri = @lift clamp.(data["truth_Ri"][:,$frame], -1, 2)
    test_Ri = @lift clamp.(data["test_Ri"][:,$frame], -1, 2)
    test_Ri_modified_pacanowski_philander = @lift clamp.(data["test_Ri_modified_pacanowski_philander"][:,$frame], -1, 2)
    # test_Ri_NN_only = @lift clamp.(data["test_Ri_NN_only"][:,$frame], -1, 2)

    losses = data["losses"]
    losses_gradient = data["losses_gradient"]
    losses_modified_pacanowski_philander = data["losses_modified_pacanowski_philander"]
    losses_modified_pacanowski_philander_gradient = data["losses_modified_pacanowski_philander_gradient"]

    losses .= losses .+ (losses .== 0) .* eps(Float32)
    losses_gradient .= losses_gradient .+ (losses_gradient .== 0) .* eps(Float32)
    losses_modified_pacanowski_philander .= losses_modified_pacanowski_philander .+ (
                                                 losses_modified_pacanowski_philander .== 0) .* eps(Float32)
    losses_modified_pacanowski_philander_gradient .= losses_modified_pacanowski_philander_gradient .+ (
                                                          losses_modified_pacanowski_philander_gradient .== 0) .* eps(Float32)

    loss_point = @lift [losses[$frame]]

    loss_gradient_point = @lift [losses_gradient[$frame]]
    loss_modified_pacanowski_philander_point = @lift [losses_modified_pacanowski_philander[$frame]]
    loss_modified_pacanowski_philander_gradient_point = @lift [losses_modified_pacanowski_philander_gradient[$frame]]

    u_max = maximum([maximum(data["truth_u"]), maximum(data["test_u"]), maximum(data["test_u_modified_pacanowski_philander"])])
    u_min = minimum([minimum(data["truth_u"]), minimum(data["test_u"]), minimum(data["test_u_modified_pacanowski_philander"])])

    v_max = maximum([maximum(data["truth_v"]), maximum(data["test_v"]), maximum(data["test_v_modified_pacanowski_philander"])])
    v_min = minimum([minimum(data["truth_v"]), minimum(data["test_v"]), minimum(data["test_v_modified_pacanowski_philander"])])

    T_max = maximum([maximum(data["truth_T"]), maximum(data["test_T"]), maximum(data["test_T_modified_pacanowski_philander"])])
    T_min = minimum([minimum(data["truth_T"]), minimum(data["test_T"]), minimum(data["test_T_modified_pacanowski_philander"])])

    uw_max = maximum([maximum(data["truth_uw"]), maximum(data["test_uw"]), maximum(data["test_uw_modified_pacanowski_philander"]), maximum(data["test_uw_NN_only"])])
    uw_min = minimum([minimum(data["truth_uw"]), minimum(data["test_uw"]), minimum(data["test_uw_modified_pacanowski_philander"]), minimum(data["test_uw_NN_only"])])

    vw_max = maximum([maximum(data["truth_vw"]), maximum(data["test_vw"]), maximum(data["test_vw_modified_pacanowski_philander"]), maximum(data["test_vw_NN_only"])])
    vw_min = minimum([minimum(data["truth_vw"]), minimum(data["test_vw"]), minimum(data["test_vw_modified_pacanowski_philander"]), minimum(data["test_vw_NN_only"])])
    
    wT_max = maximum([maximum(data["truth_wT"]), maximum(data["test_wT"]), maximum(data["test_wT_modified_pacanowski_philander"]), maximum(data["test_wT_NN_only"])])
    wT_min = minimum([minimum(data["truth_wT"]), minimum(data["test_wT"]), minimum(data["test_wT_modified_pacanowski_philander"]), minimum(data["test_wT_NN_only"])])

    losses_max = maximum([maximum(data["losses"]), maximum(data["losses_gradient"]), maximum(data["losses_modified_pacanowski_philander"]), maximum(data["losses_modified_pacanowski_philander_gradient"])])
    losses_min = minimum([minimum(data["losses"][2:end]), minimum(data["losses_gradient"][2:end]), minimum(data["losses_modified_pacanowski_philander"][2:end]), minimum(data["losses_modified_pacanowski_philander_gradient"][2:end])])

    train_parameters = data["train_parameters"]
    ν₀ = train_parameters.ν₀
    ν₋ = train_parameters.ν₋
    ΔRi = train_parameters.ΔRi
    Riᶜ = train_parameters.Riᶜ
    Pr = train_parameters.Pr
    gradient_scaling = train_parameters.gradient_scaling

    BC_str = @sprintf "Momentum Flux = %.1e m² s⁻², Buoyancy Flux = %.1e m² s⁻³" data["truth_uw"][end, 1] data["truth_wT"][end, 1]
    plot_title = @lift "$animation_type Data: $BC_str, Time = $(round(times[$frame], digits=2)) days"

    diffusivity_str = @sprintf "ν₀ = %.1e m² s⁻¹, ν₋ = %.1e m² s⁻¹, ΔRi = %.1e, Riᶜ = %.2f, Pr=%.1f" ν₀ ν₋ ΔRi Riᶜ Pr 

    scaling_str = @sprintf "Gradient Scaling = %.1e" gradient_scaling

    plot_subtitle = "$n_trainings Training Simulations ($training_types): $diffusivity_str, $scaling_str"
    
    fig = Figure(resolution=(1920, 1080))
    color_palette = distinguishable_colors(9, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    colors = (truth=color_palette[1], 
              test=color_palette[2], 
              test_modified_pacanoswki_philander=color_palette[3],
              test_NN_only=color_palette[4])
    colors_losses = (loss=color_palette[5],
                     loss_modified_pacanowski_philander=color_palette[6],
                     loss_gradient=color_palette[7],
                     loss_gradient_modified_pacanowski_philander=color_palette[8],
                     point=color_palette[9])

    u_str = "u / m s⁻¹"
    v_str = "v / m s⁻¹"
    T_str = "T / °C"
    uw_str = "uw / m² s⁻²"
    vw_str = "vw / m² s⁻²"
    wT_str = "wT / m s⁻¹ °C"

    zc = data["depth_profile"]
    zf = data["depth_flux"]
    zf_interior = zf[2:end-1]
    z_str = "z / m"

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, truth_u, zc, linewidth=3, color=colors.truth), 
                lines!(ax_u, test_u_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_u, test_u, zc, linewidth=3, color=colors.test)
                ]
                # lines!(ax_u, test_u_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, truth_v, zc, linewidth=3, color=colors.truth), 
                lines!(ax_v, test_v_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_v, test_v, zc, linewidth=3, color=colors.test),
                ]
                # lines!(ax_v, test_v_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, truth_T, zc, linewidth=3, color=colors.truth), 
                lines!(ax_T, test_T_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_T, test_T, zc, linewidth=3, color=colors.test)
                ]
                # lines!(ax_T, test_T_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, truth_uw, zf, linewidth=3, color=colors.truth), 
                lines!(ax_uw, test_uw_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_uw, test_uw_NN_only, zf_interior, linewidth=3, color=colors.test_NN_only),
                lines!(ax_uw, test_uw, zf, linewidth=3, color=colors.test), 
                ]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, truth_vw, zf, linewidth=3, color=colors.truth), 
                lines!(ax_vw, test_vw_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_vw, test_vw_NN_only, zf_interior, linewidth=3, color=colors.test_NN_only),
                lines!(ax_vw, test_vw, zf, linewidth=3, color=colors.test)]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, truth_wT, zf, linewidth=3, color=colors.truth), 
                lines!(ax_wT, test_wT_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_wT, test_wT_NN_only, zf_interior, linewidth=3, color=colors.test_NN_only),
                lines!(ax_wT, test_wT, zf, linewidth=3, color=colors.test)]
                CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[1, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, truth_Ri, zf, linewidth=3, color=colors.truth), 
                lines!(ax_Ri, test_Ri_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_Ri, test_Ri, zf, linewidth=3, color=colors.test)]

    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    ax_losses = fig[2, 4] = Axis(fig, xlabel="Time / days", ylabel="Loss", yscale=CairoMakie.log10)
    losses_lines = [lines!(ax_losses, times, losses, linewidth=3, color=colors_losses.loss),
                    lines!(ax_losses, times, losses_modified_pacanowski_philander, linewidth=3, color=colors_losses.loss_modified_pacanowski_philander),
                    lines!(ax_losses, times, losses_gradient, linewidth=3, color=colors_losses.loss_gradient),
                    lines!(ax_losses, times, losses_modified_pacanowski_philander_gradient, linewidth=3, color=colors_losses.loss_gradient_modified_pacanowski_philander)]
    losses_point = [CairoMakie.scatter!(ax_losses, time_point, loss_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_gradient_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_modified_pacanowski_philander_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_modified_pacanowski_philander_gradient_point, color=colors_losses.point)]                
    
    CairoMakie.xlims!(ax_losses, times[1], times[end])
    CairoMakie.ylims!(ax_losses, losses_min, losses_max)

    legend = fig[1, 5] = Legend(fig, uw_lines, ["Oceananigans.jl LES", 
                                                "Modified Pac-Phil Only", 
                                                "NN Only",
                                                "NN + Modified Pac-Phil"])
    legend = fig[2, 5] = Legend(fig, losses_lines, ["Profile Loss, NN + Modified Pac-Phil", 
                                                    "Profile Loss, Modified Pac-Phil Only", 
                                                    "Gradient Loss, NN + Modified Pac-Phil", 
                                                    "Gradient Loss, Modified Pac-Phil Only"])
    # legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NN + Modified Pac-Phil", "Modified Pac-Phil Only"])
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
    all_data = [WindMixing.data(train_file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true) for train_file in train_files]

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

function animate_training_results(test_files, FILE_NAME; trange=1:1:1153, fps=30, gif=false, mp4=true)
    DATA_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
    OUTPUT_PATH = joinpath(pwd(), "NDE_output_diffeq", FILE_NAME)

    if !ispath(OUTPUT_PATH)
        mkdir(OUTPUT_PATH)
    end

    @info "Loading Data"
    file = jldopen(DATA_PATH, "r")
    losses = file["losses"]
    @info "Training Loss = $(minimum(losses))"
    train_files = file["training_info/train_files"]
    train_parameters = file["training_info/parameters"]
    uw_NN = file["neural_network/uw"]
    vw_NN = file["neural_network/vw"]
    wT_NN = file["neural_network/wT"]
    close(file)

    @info "Loading Training Data"
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
    training_types = generate_training_types_str(FILE_NAME)

    for test_file in test_files
        @info "Generating Data: $test_file"
        𝒟test = WindMixing.data(test_file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

        @info "Solving NDE: $test_file"
        plot_data = NDE_profile_mutating(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange,
                                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                                ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], 
                                Riᶜ=train_parameters["Riᶜ"], convective_adjustment=train_parameters["convective_adjustment"],
                                smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"],
                                zero_weights=train_parameters["zero_weights"],
                                gradient_scaling=train_parameters["gradient_scaling"])
        
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

        VIDEO_PATH = joinpath(OUTPUT_PATH, "$VIDEO_NAME")

        @info "Animating $test_file Video"
        animate_profiles_fluxes_comparison(plot_data, VIDEO_PATH, fps=fps, gif=gif, mp4=mp4, 
                                                animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
        @info "$test_file Animation Completed"
    end

    @info "Plotting Loss..."
    Plots.plot(1:1:length(losses), losses, yscale=:log10)
    Plots.xlabel!("Iteration")
    Plots.ylabel!("Loss mse")
    savefig(joinpath(OUTPUT_PATH, "loss.pdf"))
end

function animate_training_results_oceananigans(test_files, timestep, FILE_NAME, OUTPUT_DIR)
    EXTRACTED_FILE_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
    extracted_training_file = jldopen(EXTRACTED_FILE_PATH)
    losses = extracted_training_file["losses"]
    solve_oceananigans_modified_pacanowski_philander_nn(test_files, EXTRACTED_FILE_PATH, OUTPUT_DIR, timestep=timestep)

    train_files = extracted_training_file["training_info/train_files"]
    train_parameters = extracted_training_file["training_info/parameters"]

    close(extracted_training_file)

    for test_file in test_files
        @info "Processing $test_file solution"
        if test_file in train_files
            SOL_DIR = joinpath(OUTPUT_DIR, "train_$test_file")
            animation_type = "Training"
        else
            SOL_DIR = joinpath(OUTPUT_DIR, "test_$test_file")
            animation_type = "Testing"
        end

        plot_data = NDE_profile_oceananigans(SOL_DIR, train_files, [test_file],
                                        ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], 
                                        ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], Pr=train_parameters["Pr"], 
                                        gradient_scaling=train_parameters["gradient_scaling"],
                                        OUTPUT_PATH=joinpath(SOL_DIR, "profiles_fluxes.jld2"))

        n_trainings = length(train_files)
        training_types = generate_training_types_str(FILE_NAME)
        VIDEO_NAME = "profiles_fluxes_comparison_animation"

        @info "Animating $test_file solution"
        animate_profiles_fluxes_comparison(plot_data, joinpath(SOL_DIR, VIDEO_NAME), fps=30, 
                                                        animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
    end

    @info "Plotting Loss..."
    Plots.plot(1:1:length(losses), losses, yscale=:log10)
    Plots.xlabel!("Iteration")
    Plots.ylabel!("Loss mse")
    savefig(joinpath(OUTPUT_DIR, "loss.pdf"))
end