function plot_LES_3D(frame, FILE_DIR, OUTPUT_PATH, axis_images; title="", colorscheme=:ice, rev=false)
    xy_file = jldopen(joinpath(FILE_DIR, "xy_slice.jld2"))
    xz_file = jldopen(joinpath(FILE_DIR, "xz_slice.jld2"))
    yz_file = jldopen(joinpath(FILE_DIR, "yz_slice.jld2"))
    instantaneous_statistics = jldopen(joinpath(FILE_DIR, "instantaneous_statistics.jld2"))

    frames = keys(xy_file["timeseries/T"])
    iteration = frames[frame]

    color_range_max = maximum(maximum.([xz_file["timeseries/T/$iteration"], yz_file["timeseries/T/$iteration"], xy_file["timeseries/T/$iteration"]]))
    color_range_min = minimum(minimum.([xz_file["timeseries/T/$iteration"], yz_file["timeseries/T/$iteration"], xy_file["timeseries/T/$iteration"]]))
    color_range = (color_range_min, color_range_max)

    xC = xz_file["grid/xC"][4:end-3]
    yC = xz_file["grid/yC"][4:end-3]
    zC = xz_file["grid/zC"][4:end-3]

    colormap = cgrad(colorscheme, scale=:log10, rev=rev)

    fig = Figure(resolution=(1920, 1080))
    ax = fig[1,1] = CairoMakie.Axis3(fig, aspect=(1, 1, 0.5), xlabel="x /m", ylabel="y /m", zlabel="z /m")

    x_xz = fill(xC[1], 128)
    y_xz = yC
    z_xz = zeros(length(x_xz), length(y_xz))

    for i in 1:size(z_xz, 1)
        z_xz[i,:] .= zC[i]
    end

    z_xz
    T_xz = transpose(hcat([xz_file["timeseries/T/$iteration"][:, :, i] for i in 1:length(zC)]... ))

    xz_surface = CairoMakie.surface!(ax, x_xz, y_xz, z_xz, color=T_xz, colormap=colormap)

    x_yz = xC
    y_yz = fill(yC[1], 128)
    z_yz = zeros(length(x_yz), length(y_yz))

    for i in 1:size(z_yz, 2)
        z_yz[:,i] .= zC[i]
    end

    z_yz

    T_yz = similar(z_yz)

    for i in 1:size(T_yz, 2)
        T_yz[:,i] = yz_file["timeseries/T/$iteration"][:, :, i]
    end
    T_yz

    yz_surface = CairoMakie.surface!(ax, x_yz, y_yz, z_yz, color=T_yz, colormap=colormap)

    x_xy = xC
    y_xy = yC
    z_xy = zeros(length(y_xy), length(x_xy))

    T_xy = xy_file["timeseries/T/$iteration"][:,:,1]

    xy_surface = CairoMakie.surface!(ax, x_xy, y_xy, z_xy, color=T_xy, colormap=colormap)

    xz_surface.colorrange = color_range
    yz_surface.colorrange = color_range
    xy_surface.colorrange = color_range

    ax_T_3D = fig[2,1] = CairoMakie.Axis(fig, aspect=DataAspect())

    rel_size = 40
    aspect = 1 / 4

    hidedecorations!(ax_T_3D)
    hidespines!(ax_T_3D)
    image!(ax_T_3D, axis_images.T_3D)
    rowsize!(fig.layout, 2, CairoMakie.Relative(1 / rel_size))

    colorbar = CairoMakie.Colorbar(fig[3, 1], xy_surface, vertical=false)

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

    label_a = fig[1, 1, TopLeft()] = Label(fig, "a)", textsize = 20, halign = :right)
    label_b = plots_sublayout[1,1, Top()] = Label(fig, "b)", textsize = 20, halign = :right)
    label_c = plots_sublayout[3,1, Top()] = Label(fig, "c)", textsize = 20, halign = :right)
    label_d = plots_sublayout[5,1, Top()] = Label(fig, "d)", textsize = 20, halign = :right)

    u_line = CairoMakie.lines!(ax_u, instantaneous_statistics["timeseries/u/$iteration"][:], zC)
    v_line = CairoMakie.lines!(ax_v, instantaneous_statistics["timeseries/v/$iteration"][:], zC)
    T_line = CairoMakie.lines!(ax_T, instantaneous_statistics["timeseries/T/$iteration"][:], zC)

    if title !== ""
        supertitle = fig[0, :] = Label(fig, title, textsize=25)
    end

    trim!(fig.layout)

    close(xy_file)
    close(xz_file)
    close(yz_file)

    save(OUTPUT_PATH, fig, px_per_unit=4, pt_per_unit=2)
end

function plot_T_profiles_final(NDE_profiles, frame, subplot_titles, axis_images, FILE_PATH)
    T_datasets = [
        [
            data["truth_T"][:,frame],
            data["test_T_modified_pacanowski_philander"][:,frame],
            data["test_T_kpp"][:,frame],
            data["test_T"][:,frame],
        ] for data in NDE_profiles
    ]

    @inline function find_lims(datasets)
        return maximum(maximum([maximum.(data) for data in datasets])), minimum(minimum([minimum.(data) for data in datasets]))
    end

    T_max, T_min = find_lims(T_datasets)

    fig = Figure(resolution=(1280, 914))
    
    colors = distinguishable_colors(length(T_datasets[1])+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    temp_color = colors[2]
    colors[2] = colors[4]
    colors[4] = temp_color

    T_img = axis_images.T
    z_img = axis_images.z

    zc = NDE_profiles[1]["depth_profile"]

    rel_size = 30
    # aspect = 1920 / 1080
    aspect = 2

    ax_T₁ = fig[2,2] = CairoMakie.Axis(fig)
    ax_T₂ = fig[2,3] = CairoMakie.Axis(fig)
    ax_T₃ = fig[2,4] = CairoMakie.Axis(fig)
    ax_T₄ = fig[5,2] = CairoMakie.Axis(fig)
    ax_T₅ = fig[5,3] = CairoMakie.Axis(fig)
    ax_T₆ = fig[5,4] = CairoMakie.Axis(fig)

    axs = [
        ax_T₁
        ax_T₂
        ax_T₃
        ax_T₄
        ax_T₅
        ax_T₆
    ]

    y_ax_T₁ = fig[2,1] = CairoMakie.Axis(fig, aspect=DataAspect())
    # y_ax_T₂ = fig[2,3] = CairoMakie.Axis(fig, aspect=DataAspect())
    # y_ax_T₃ = fig[2,5] = CairoMakie.Axis(fig, aspect=DataAspect())
    y_ax_T₄ = fig[5,1] = CairoMakie.Axis(fig, aspect=DataAspect())
    # y_ax_T₅ = fig[5,3] = CairoMakie.Axis(fig, aspect=DataAspect())
    # y_ax_T₆ = fig[5,5] = CairoMakie.Axis(fig, aspect=DataAspect())

    x_ax_T₁ = fig[3,2] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₂ = fig[3,3] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₃ = fig[3,4] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₄ = fig[6,2] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₅ = fig[6,3] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₆ = fig[6,4] = CairoMakie.Axis(fig, aspect=DataAspect())

    title_size = 16
    ax_title₁ = fig[1,2] = Label(fig, subplot_titles[1], textsize=title_size)
    ax_title₂ = fig[1,3] = Label(fig, subplot_titles[2], textsize=title_size)
    ax_title₃ = fig[1,4] = Label(fig, subplot_titles[3], textsize=title_size)
    ax_title₄ = fig[4,2] = Label(fig, subplot_titles[4], textsize=title_size)
    ax_title₅ = fig[4,3] = Label(fig, subplot_titles[5], textsize=title_size)
    ax_title₆ = fig[4,4] = Label(fig, subplot_titles[6], textsize=title_size)

    x_axs = [
        x_ax_T₁
        x_ax_T₂
        x_ax_T₃
        x_ax_T₄
        x_ax_T₅
        x_ax_T₆
    ]

    y_axs = [
        y_ax_T₁
        # y_ax_T₂
        # y_ax_T₃
        y_ax_T₄
        # y_ax_T₅
        # y_ax_T₆
    ]

    title_axs = [
        ax_title₁
        ax_title₂
        ax_title₃
        ax_title₄
        ax_title₅
        ax_title₆
    ]

    for i in 1:length(x_axs)
        hidedecorations!(x_axs[i])
        hidespines!(x_axs[i])
    end

    for y_ax in y_axs
        hidedecorations!(y_ax)
        hidespines!(y_ax)
    end


    for ax in x_axs
        image!(ax, axis_images.T)
    end

    for ax in y_axs
        image!(ax, axis_images.z)
    end

    for ax in axs
        # CairoMakie.xlims!(ax, T_min, T_max)
        CairoMakie.ylims!(ax, minimum(zc), 0)
    end

    
    linkyaxes!(axs...)
    hideydecorations!(axs[2], grid = false)
    hideydecorations!(axs[3], grid = false)
    hideydecorations!(axs[5], grid = false)
    hideydecorations!(axs[6], grid = false)

    rowsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size))
    rowsize!(fig.layout, 6, CairoMakie.Relative(1 / rel_size))

    colsize!(fig.layout, 1, CairoMakie.Relative(1 / rel_size / aspect))

    colsize!(fig.layout, 2, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
    colsize!(fig.layout, 3, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
    colsize!(fig.layout, 4, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))

    # # colsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size / aspect))
    # # colsize!(fig.layout, 5, CairoMakie.Relative(1 / rel_size / aspect))

    colgap!(fig.layout, Relative(1 / rel_size / aspect / 2))
    rowgap!(fig.layout, Relative(1 / rel_size / aspect / 3))

    label_a = fig[1, 2, TopLeft()] = Label(fig, "a)", textsize = 14, halign = :right)
    label_b = fig[1, 3, TopLeft()] = Label(fig, "b)", textsize = 14, halign = :right)
    label_c = fig[1, 4, TopLeft()] = Label(fig, "c)", textsize = 14, halign = :right)
    label_d = fig[4, 2, TopLeft()] = Label(fig, "d)", textsize = 14, halign = :right)
    label_e = fig[4, 3, TopLeft()] = Label(fig, "e)", textsize = 14, halign = :right)
    label_f = fig[4, 4, TopLeft()] = Label(fig, "f)", textsize = 14, halign = :right)


    alpha = 0.4
    truth_linewidth = 7
    linewidth = 3
    
    @inline function make_lines(ax, data)
        lines = [
            lines!(ax, NDE_profiles[1]["truth_T"][:,1], zc, linestyle=:dot, color=colors[end], linewidth=linewidth);
            lines!(ax, data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
            [lines!(ax, data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(data)]
                ]
        return lines
    end

    T_lines_axs = [
        make_lines(axs[i], T_datasets[i]) for i in 1:length(T_datasets)
    ]

    legend = fig[7, :] = CairoMakie.Legend(fig, T_lines_axs[1],
            ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"],
            orientation = :horizontal
    )

    legend.tellheight = true

    trim!(fig.layout)
    save(FILE_PATH, fig, px_per_unit=4, pt_per_unit=4)
end

function plot_T_profiles_final_diurnal(NDE_profiles, frame, subplot_titles, axis_images, FILE_PATH)
    T_datasets = [
        [
            data["truth_T"][:,frame],
            data["test_T_modified_pacanowski_philander"][:,frame],
            data["test_T_kpp"][:,frame],
            data["test_T"][:,frame],
        ] for data in NDE_profiles
    ]

    @inline function find_lims(datasets)
        return maximum(maximum([maximum.(data) for data in datasets])), minimum(minimum([minimum.(data) for data in datasets]))
    end

    T_max, T_min = find_lims(T_datasets)

    fig = Figure(resolution=(1280, 500))
    
    colors = distinguishable_colors(length(T_datasets[1])+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    temp_color = colors[2]
    colors[2] = colors[4]
    colors[4] = temp_color

    T_img = axis_images.T
    z_img = axis_images.z

    zc = NDE_profiles[1]["depth_profile"]

    rel_size = 30
    # aspect = 1920 / 1080
    aspect = 2

    ax_T₁ = fig[2,2] = CairoMakie.Axis(fig)
    ax_T₂ = fig[2,3] = CairoMakie.Axis(fig)
    ax_T₃ = fig[2,4] = CairoMakie.Axis(fig)

    axs = [
        ax_T₁
        ax_T₂
        ax_T₃
    ]

    y_ax_T₁ = fig[2,1] = CairoMakie.Axis(fig, aspect=DataAspect())

    x_ax_T₁ = fig[3,2] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₂ = fig[3,3] = CairoMakie.Axis(fig, aspect=DataAspect())
    x_ax_T₃ = fig[3,4] = CairoMakie.Axis(fig, aspect=DataAspect())

    title_size = 16
    ax_title₁ = fig[1,2] = Label(fig, subplot_titles[1], textsize=title_size)
    ax_title₂ = fig[1,3] = Label(fig, subplot_titles[2], textsize=title_size)
    ax_title₃ = fig[1,4] = Label(fig, subplot_titles[3], textsize=title_size)

    x_axs = [
        x_ax_T₁
        x_ax_T₂
        x_ax_T₃
    ]

    y_axs = [
        y_ax_T₁
    ]

    title_axs = [
        ax_title₁
        ax_title₂
        ax_title₃
    ]

    for i in 1:length(x_axs)
        hidedecorations!(x_axs[i])
        hidespines!(x_axs[i])
    end

    for y_ax in y_axs
        hidedecorations!(y_ax)
        hidespines!(y_ax)
    end


    for ax in x_axs
        image!(ax, axis_images.T)
    end

    for ax in y_axs
        image!(ax, axis_images.z)
    end

    for ax in axs
        # CairoMakie.xlims!(ax, T_min, T_max)
        CairoMakie.ylims!(ax, minimum(zc), 0)
    end

    
    linkyaxes!(axs...)
    hideydecorations!(axs[2], grid = false)
    hideydecorations!(axs[3], grid = false)

    rowsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size * 1.5))

    colsize!(fig.layout, 1, CairoMakie.Relative(1 / rel_size / aspect))

    colsize!(fig.layout, 2, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
    colsize!(fig.layout, 3, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))
    colsize!(fig.layout, 4, CairoMakie.Relative((1 - 1 / rel_size / aspect) / 3))

    # # colsize!(fig.layout, 3, CairoMakie.Relative(1 / rel_size / aspect))
    # # colsize!(fig.layout, 5, CairoMakie.Relative(1 / rel_size / aspect))

    colgap!(fig.layout, Relative(1 / rel_size / aspect / 2))
    rowgap!(fig.layout, Relative(1 / rel_size / aspect / 3))

    label_a = fig[1, 2, TopLeft()] = Label(fig, "a)", textsize = 14, halign = :right)
    label_b = fig[1, 3, TopLeft()] = Label(fig, "b)", textsize = 14, halign = :right)
    label_c = fig[1, 4, TopLeft()] = Label(fig, "c)", textsize = 14, halign = :right)

    alpha = 0.4
    truth_linewidth = 7
    linewidth = 3
    
    @inline function make_lines(ax, data)
        lines = [
            lines!(ax, NDE_profiles[1]["truth_T"][:,1], zc, linestyle=:dot, color=colors[end], linewidth=linewidth);
            lines!(ax, data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
            [lines!(ax, data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(data)]
                ]
        return lines
    end

    T_lines_axs = [
        make_lines(axs[i], T_datasets[i]) for i in 1:length(T_datasets)
    ]

    legend = fig[4, :] = CairoMakie.Legend(fig, T_lines_axs[1],
            ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"],
            orientation = :horizontal
    )

    legend.tellheight = true

    trim!(fig.layout)
    save(FILE_PATH, fig, px_per_unit=4, pt_per_unit=4)
end

function plot_profiles_fluxes_final(data, frame, axis_images, FILE_PATH)
    u_data = [
        data["truth_u"][:,frame],
        data["test_u_modified_pacanowski_philander"][:,frame],
        data["test_u_kpp"][:,frame],
        data["test_u"][:,frame],
    ]

    v_data = [
        data["truth_v"][:,frame],
        data["test_v_modified_pacanowski_philander"][:,frame],
        data["test_v_kpp"][:,frame],
        data["test_v"][:,frame],
    ]

    T_data = [
        data["truth_T"][:,frame],
        data["test_T_modified_pacanowski_philander"][:,frame],
        data["test_T_kpp"][:,frame],
        data["test_T"][:,frame],
    ]
    
    uw_data = [
        data["truth_uw"][:,frame],
        data["test_uw_modified_pacanowski_philander"][:,frame],
        data["test_uw_kpp"][:,frame],
        data["test_uw"][:,frame],
    ]

    vw_data = [
        data["truth_vw"][:,frame],
        data["test_vw_modified_pacanowski_philander"][:,frame],
        data["test_vw_kpp"][:,frame],
        data["test_vw"][:,frame],
    ]

    wT_data = [
        data["truth_wT"][:,frame],
        data["test_wT_modified_pacanowski_philander"][:,frame],
        data["test_wT_kpp"][:,frame],
        data["test_wT"][:,frame],
    ]

    uw_data .*= 1f4
    vw_data .*= 1f4
    wT_data .*= 1f5

    Ri_data = [
        clamp.(data["truth_Ri"][:,frame], -1, 2),
        clamp.(data["test_Ri_modified_pacanowski_philander"][:,frame], -1, 2),
        clamp.(data["test_Ri_kpp"][:,frame], -1, 2),
        clamp.(data["test_Ri"][:,frame], -1, 2),
    ]

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

    # fig = Figure(resolution=(1920, 1080))
    fig = Figure(resolution=(2000, 960))
    
    # colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    colors = distinguishable_colors(length(uw_data)+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    temp_color = colors[2]
    colors[2] = colors[4]
    colors[4] = temp_color

    zc = data["depth_profile"]
    zf = data["depth_flux"]

    rel_size = 40
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

    alpha=0.4
    truth_linewidth = 7
    linewidth = 3
    
    # CairoMakie.xlims!(ax_u, u_min, u_max)
    # CairoMakie.xlims!(ax_v, v_min, v_max)
    # CairoMakie.xlims!(ax_T, T_min, T_max)
    # CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    # CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    # CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.xlims!(ax_Ri, -1, 2)

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    label_a = fig[1, 1, Top()] = Label(fig, "a)", textsize = 20, halign = :right)
    label_b = fig[1, 3, Top()] = Label(fig, "b)", textsize = 20, halign = :right)
    label_d = fig[1, 6, Top()] = Label(fig, "d)", textsize = 20, halign = :right)
    label_e = fig[3, 1, Top()] = Label(fig, "e)", textsize = 20, halign = :right)
    label_f = fig[3, 3, Top()] = Label(fig, "f)", textsize = 20, halign = :right)
    label_g = fig[3, 6, Top()] = Label(fig, "g)", textsize = 20, halign = :right)
    
    label_c = T_layout[1, 1, Top()] = Label(fig, "c)", textsize = 20, halign = :right)

    u_lines = [
            lines!(ax_u, u_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_u, u_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(u_data)]
    ]

    v_lines = [
            lines!(ax_v, v_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_v, v_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(v_data)]
    ]

    T_lines = [
            lines!(ax_T, data["truth_T"][:,1], zc, linewidth=linewidth, color=colors[end], linestyle=:dot)
            lines!(ax_T, T_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_T, T_data[i], zc, linewidth=linewidth, color=colors[i]) for i in 2:length(T_data)]
    ]

    uw_lines = [
            lines!(ax_uw, uw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_uw, uw_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(uw_data)]
    ]

    vw_lines = [
            lines!(ax_vw, vw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_vw, vw_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(vw_data)]
    ]

    wT_lines = [
        lines!(ax_wT, wT_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_wT, wT_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(wT_data)]
    ]

    Ri_lines = [
            lines!(ax_Ri, Ri_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha));
        [lines!(ax_Ri, Ri_data[i], zf, linewidth=linewidth, color=colors[i]) for i in 2:length(Ri_data)]
    ]

    axislegend(ax_T, T_lines, ["Initial Stratification", "Large Eddy Simulation", "Ri-based Diffusivity Only", "K-Profile Parameterisation", "NN Embedded in Oceananigans.jl"], "Data Type", position = :rb)
    
    trim!(fig.layout)
    
    save(FILE_PATH, fig, px_per_unit=4, pt_per_unit=4)
end

function plot_profiles_fluxes_final_LES(data, frame, axis_images, FILE_PATH)
    u_data = [
        data["truth_u"][:,frame],
    ]

    v_data = [
        data["truth_v"][:,frame],
    ]

    T_data = [
        data["truth_T"][:,frame],
    ]
    
    uw_data = [
        data["truth_uw"][:,frame],
    ]

    vw_data = [
        data["truth_vw"][:,frame],
    ]

    wT_data = [
        data["truth_wT"][:,frame],
    ]

    uw_data .*= 1f4
    vw_data .*= 1f4
    wT_data .*= 1f5

    Ri_data = [
        clamp.(data["truth_Ri"][:,frame], -1, 2),
    ]

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

    fig = Figure(resolution=(2000, 960))
    
    colors = distinguishable_colors(length(uw_data), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    # temp_color = colors[2]
    # colors[2] = colors[4]
    # colors[4] = temp_color

    zc = data["depth_profile"]
    zf = data["depth_flux"]

    rel_size = 40
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

    alpha = 0.5
    truth_linewidth = 7
    linewidth = 3
    
    # CairoMakie.xlims!(ax_u, u_min, u_max)
    # CairoMakie.xlims!(ax_v, v_min, v_max)
    # CairoMakie.xlims!(ax_T, T_min, T_max)
    # CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    # CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    # CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.xlims!(ax_Ri, -1, 2)

    CairoMakie.ylims!(ax_u, minimum(zc), 0)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    label_a = fig[1, 1, Top()] = Label(fig, "a)", textsize = 20, halign = :right)
    label_b = fig[1, 3, Top()] = Label(fig, "b)", textsize = 20, halign = :right)
    label_d = fig[1, 6, Top()] = Label(fig, "d)", textsize = 20, halign = :right)
    label_e = fig[3, 1, Top()] = Label(fig, "e)", textsize = 20, halign = :right)
    label_f = fig[3, 3, Top()] = Label(fig, "f)", textsize = 20, halign = :right)
    label_g = fig[3, 6, Top()] = Label(fig, "g)", textsize = 20, halign = :right)
    
    label_c = T_layout[1, 1, Top()] = Label(fig, "c)", textsize = 20, halign = :right)

    u_lines = [
            lines!(ax_u, u_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    v_lines = [
            lines!(ax_v, v_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    T_lines = [
            lines!(ax_T, T_data[1], zc, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    uw_lines = [
            lines!(ax_uw, uw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    vw_lines = [
            lines!(ax_vw, vw_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    wT_lines = [
        lines!(ax_wT, wT_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    Ri_lines = [
            lines!(ax_Ri, Ri_data[1], zf, linewidth=truth_linewidth, color=(colors[1], alpha))
    ]

    axislegend(ax_T, T_lines, ["Oceananigans.jl Large Eddy Simulation"], "Data Type", position = :rb)
    
    trim!(fig.layout)
    
    save(FILE_PATH, fig, px_per_unit=4, pt_per_unit=4)
end