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

    plots_sublayout = fig[:,2] = GridLayout()

    colsize!(fig.layout, 2, CairoMakie.Relative(aspect))
    rowgap!(fig.layout, Relative(1 / 50 / 3))

    colorbar = CairoMakie.Colorbar(fig[3, 1], xy_surface, vertical=false)

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