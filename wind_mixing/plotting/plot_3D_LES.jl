using CairoMakie
using JLD2

FILE_DIR = "../Data/three_layer_constant_fluxes_linear_hr144_Qu2.2e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling"
frame = 145
colorscheme = :turbo

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

colormap = cgrad(colorscheme, scale=:log10, rev=false)

##
fig = Figure(resolution=(2000, 2000), fontsize=50, figure_padding = 50)

ga = fig[1:3,1:3] = GridLayout()
gb = fig[4,1] = GridLayout()
gc = fig[4,2] = GridLayout()
gd = fig[4,3] = GridLayout()

ax = CairoMakie.Axis3(ga[1,1], aspect=(1, 1, 0.5), xlabel=L"x (m) $ $", ylabel=L"y (m) $ $", zlabel=L"z (m) $ $", 
                                     xlabeloffset=100, zlabeloffset=150, ylabeloffset=100)

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
colorbar = CairoMakie.Colorbar(ga[1, 2], xy_surface, label=L"$T$ ($\degree$C)")

colorbar.alignmode = Mixed(right = 0)

x_xy = xC
y_xy = yC
z_xy = zeros(length(y_xy), length(x_xy))

T_xy = xy_file["timeseries/T/$iteration"][:,:,1]

xy_surface = CairoMakie.surface!(ax, x_xy, y_xy, z_xy, color=T_xy, colormap=colormap)

xz_surface.colorrange = color_range
yz_surface.colorrange = color_range
xy_surface.colorrange = color_range

ax_u = CairoMakie.Axis(gb[1,1], ylabel=L"z (m) $ $", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
ax_v = CairoMakie.Axis(gc[1,1], ylabel=L"z (m) $ $", xlabel=L"$\overline{v}$ (m s$^{-1}$)")
ax_T = CairoMakie.Axis(gd[1,1], ylabel=L"z (m) $ $", xlabel=L"$\overline{T}$ ($\degree$C)")

lines!(ax_u, instantaneous_statistics["timeseries/u/$iteration"][:], zC, linewidth=10)
lines!(ax_v, instantaneous_statistics["timeseries/v/$iteration"][:], zC, linewidth=10)
lines!(ax_T, instantaneous_statistics["timeseries/T/$iteration"][:], zC, linewidth=10)

for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 50,
        font = :bold,
        padding = (0, 50, 25, 0),
        halign = :right)
end

display(fig)

##

save("plots/3D_LES.png", fig, px_per_unit=4, pt_per_unit=2)
save("plots/3D_LES.pdf", fig, px_per_unit=4, pt_per_unit=2)

##
close(xz_file)
close(xy_file)
close(yz_file)
close(instantaneous_statistics)

##
