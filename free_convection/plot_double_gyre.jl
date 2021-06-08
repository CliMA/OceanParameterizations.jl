using Printf
using CairoMakie

using Oceananigans
using Oceananigans.Units

ds = FieldDataset("double_gyre.jld2", metadata_paths=[])

u = ds["u"]
v = ds["v"]
w = ds["w"]
T = ds["T"]
# η = ds["η"]

xc = xnodes(T)
yc = ynodes(T)
zc = znodes(T)
xf = xnodes(u)
yf = ynodes(v)
zf = znodes(w)

times = T.times
Nx, Ny, Nz, Nt = size(T)

Nx½ = round(Int, Nx/2)
Ny½ = round(Int, Ny/2)

for var in (u, v, T)
    name = var.name

    frame = Node(1)
    title = @lift "Double gyre: $(prettytime(times[$frame]))"

    var_interior = interior(var)
    var_surface = @lift var_interior[:, :, Nz, $frame]
    var_meridional = @lift var_interior[:, Ny½, :, $frame]
    var_zonal = @lift var_interior[Nx½, :, :, $frame]

    x = var === u ? xf : xc
    y = var === v ? yf : yc
    z = var === w ? zf : zc

    cmap = var === T ? :thermal : :balance
    clims = var === T ? (0, 35) : (-0.25, 0.25)
    label = var === T ? "temperature (°C)" : "velocity (m/s)"

    kwargs = (colormap=cmap, levels=range(clims..., length=20), extendlow=:auto, extendhigh=:auto)

    fig = Figure()

    ax1 = fig[1, 1] = Axis(fig, title="$name(x, y, 0)", xlabel="x (km)", ylabel="y (km)")
    cf1 = contourf!(ax1, x, y, var_surface; kwargs...)
    xlims!(ax1, extrema(xf))
    ylims!(ax1, extrema(yf))

    ax2 = fig[1, 2] = Axis(fig, title="$name(x, 0, z)", xlabel="x (km)", ylabel="z (km)")
    cf2 = contourf!(ax2, x, z, var_meridional; kwargs...)
    xlims!(ax2, extrema(xf))
    ylims!(ax2, extrema(zf))

    ax3 = fig[1, 3] = Axis(fig, title="$name(0, y, z)", xlabel="y (km)", ylabel="z (km)")
    cf3 = contourf!(ax3, y, z, var_zonal; kwargs...)
    xlims!(ax3, extrema(yf))
    ylims!(ax3, extrema(zf))

    cb1 = fig[1, 4] = Colorbar(fig, cf1, label=label, width=30)

    supertitle = fig[0, :] = Label(fig, title, textsize=30)

    record(fig, "double_gyre_$name.mp4", 1:Nt, framerate=10) do n
        @info "Animating double gyre $name frame $n/$(length(times))..."
        frame[] = n
    end
end

# Nxs = round.(Int, [Nx/10, Nx/2, 9Nx/10])
# Nys = round.(Int, [Ny/10, Ny/2, 9Ny/10])

# frame = Node(1)
# fig = Figure(resolution = (1920, 1080))

# for (i, nx) in enumerate(Nxs), (j, ny) in enumerate(Nys)
#     title = @sprintf("x = %d km, y = %d km", xc[nx], yc[ny])
#     T_profile = @lift ds["T"][nx, ny, :, $frame]

#     ax = fig[i, j] = Axis(fig, title=title, xlabel="Temperature (°C)", ylabel="z (km)")
#     T_plot = lines!(ax, T_profile, zc, linewidth=3)

#     xlims!(ax, (0, 35))
#     ylims!(ax, extrema(zf))
# end

# title = @lift "Double gyre day $(round(Int, ds["time"][$frame] / 86400))"
# supertitle = fig[0, :] = Label(fig, title, textsize=30)

# record(fig, "double_gyre_T_profiles.mp4", 1:50:length(ds["time"]), framerate=20) do n
#     @info "Animating double gyre T profiles frame $n/$(length(ds["time"]))..."
#     frame[] = n
# end
