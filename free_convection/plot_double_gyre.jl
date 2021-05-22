using Printf
using NCDatasets
using CairoMakie
using Oceananigans.Utils

ds = NCDataset("double_gyre_mitgcm.nc")

xc = ds["xC"] / 1000
yc = ds["yC"] / 1000
zc = ds["zC"] / 1000
xf = ds["xF"] / 1000
yf = ds["yF"] / 1000
zf = ds["zF"] / 1000

Nx, Ny, Nz = length(xc), length(yc), length(zc)

Nx½ = round(Int, Nx/2)
Ny½ = round(Int, Ny/2)

for var in ("u", "v", "T")
    frame = Node(1)
    title = @lift "Double gyre: $(prettytime(ds["time"][$frame]))"
    var_surface = @lift ds[var][:, :, Nz, $frame]
    var_meridional = @lift ds[var][:, Ny½, :, $frame]
    var_zonal = @lift ds[var][Nx½, :, :, $frame]

    x = var == "u" ? xf : xc
    y = var == "v" ? yf : yc
    z = var == "w" ? zf : zc

    cmap = var == "T" ? :thermal : :balance
    clims = var == "T" ? (0, 35) : (-0.25, 0.25)
    label = var == "T" ? "temperature (°C)" : "velocity (m/s)"

    kwargs = (colormap=cmap, levels=range(clims..., length=20), extendlow=:auto, extendhigh=:auto)

    fig = Figure(resolution=(1920, 1080))

    ax1 = fig[1, 1] = Axis(fig, title="$var(x, y, 0)", xlabel="x (km)", ylabel="y (km)")
    cf1 = contourf!(ax1, x, y, var_surface; kwargs...)
    xlims!(ax1, extrema(xf))
    ylims!(ax1, extrema(yf))

    ax2 = fig[1, 2] = Axis(fig, title="$var(x, 0, z)", xlabel="x (km)", ylabel="z (km)")
    cf2 = contourf!(ax2, x, z, var_meridional; kwargs...)
    xlims!(ax2, extrema(xf))
    ylims!(ax2, extrema(zf))

    ax3 = fig[1, 3] = Axis(fig, title="$var(0, y, z)", xlabel="y (km)", ylabel="z (km)")
    cf3 = contourf!(ax3, y, z, var_zonal; kwargs...)
    xlims!(ax3, extrema(yf))
    ylims!(ax3, extrema(zf))

    cb1 = fig[1, 4] = Colorbar(fig, cf1, label=label, width=30)

    supertitle = fig[0, :] = Label(fig, title, textsize=30)

    record(fig, "double_gyre_$var.mp4", 1:10:length(ds["time"]), framerate=10) do n
        @info "Animating double gyre $var frame $n/$(length(ds["time"]))..."
        frame[] = n
    end
end

Nxs = round.(Int, [Nx/10, Nx/2, 9Nx/10])
Nys = round.(Int, [Ny/10, Ny/2, 9Ny/10])

frame = Node(1)
fig = Figure(resolution = (1920, 1080))

for (i, nx) in enumerate(Nxs), (j, ny) in enumerate(Nys)
    title = @sprintf("x = %d km, y = %d km", xc[nx], yc[ny])
    T_profile = @lift ds["T"][nx, ny, :, $frame]

    ax = fig[i, j] = Axis(fig, title=title, xlabel="Temperature (°C)", ylabel="z (km)")
    T_plot = lines!(ax, T_profile, zc, linewidth=3)

    xlims!(ax, (0, 35))
    ylims!(ax, extrema(zf))
end

title = @lift "Double gyre day $(round(Int, ds["time"][$frame] / 86400))"
supertitle = fig[0, :] = Label(fig, title, textsize=30)

record(fig, "double_gyre_T_profiles.mp4", 1:50:length(ds["time"]), framerate=20) do n
    @info "Animating double gyre T profiles frame $n/$(length(ds["time"]))..."
    frame[] = n
end

close(ds)
