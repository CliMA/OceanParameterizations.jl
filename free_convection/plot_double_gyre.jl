using NCDatasets
using CairoMakie

ds = NCDataset("double_gyre.nc")

xc = ds["xC"] / 1000
yc = ds["yC"] / 1000
zc = ds["zC"] / 1000
xf = ds["xF"] / 1000
yf = ds["yF"] / 1000
zf = ds["zF"] / 1000

Nx, Ny, Nz = length(xc), length(yc), length(zc)

Nx½ = round(Int, Nx/2)
Ny½ = round(Int, Ny/2)
#=
for var in ("u", "v", "b")
    frame = Node(1)
    var_surface = @lift ds[var][:, :, Nz, $frame]
    var_meridional = @lift ds[var][:, Ny½, :, $frame]
    var_zonal = @lift ds[var][Nx½, :, :, $frame]

    cmap = var == "b" ? :thermal : :balance
    clims = var == "b" ? (0, 0.12) : (-1, 1)
    label = var == "b" ? "buoyancy (m/s²)" : "velocity (m/s)"

    fig = Figure(resolution = (1920, 1080))

    ax1 = fig[1, 1] = Axis(fig, title="$var(x, y, 0)", xlabel="x (km)", ylabel="y (km)")
    hm1 = heatmap!(ax1, xc, yc, var_surface, colormap=cmap, colorrange=clims)
    # xlims!(ax1, extrema(xf))
    # ylims!(ax1, extrema(yf))

    ax2 = fig[1, 2] = Axis(fig, title="$var(x, 0, z)", xlabel="x (km)", ylabel="z (km)")
    hm2 = heatmap!(ax2, xc, zc, var_meridional, colormap=cmap, colorrange=clims)

    ax3 = fig[1, 3] = Axis(fig, title="b(0, y, z)", xlabel="y (km)", ylabel="z (km)")
    hm3 = heatmap!(ax3, yc, zc, var_zonal, colormap=cmap, colorrange=clims)

    cb1 = fig[1, 4] = Colorbar(fig, hm1, label="buoyancy (m/s²)", width=30)

    record(fig, "double_gyre_$var.mp4", 1:length(ds["time"]), framerate=30) do n
        @info "Animating double gyre $var frame $n/$(length(ds["time"]))..."
        frame[] = n
    end
end
=#
Nxs = round.(Int, [Nx/10, Nx/2, 9Nx/10])
Nys = round.(Int, [Ny/10, Ny/2, 9Ny/10])

frame = Node(1)
fig = Figure(resolution = (1920, 1080))

for (i, nx) in enumerate(Nxs), (j, ny) in enumerate(Nys)
    b_profile = @lift ds["b"][nx, ny, :, $frame]

    ax_ij = fig[i, j] = Axis(fig)
    b_ij = plot!(ax_ij, b_profile, zc)
    xlims!(ax_ij, (0, 0.12))
    ylims!(ax_ij, extrema(zf))
end

record(fig, "double_gyre_b_profiles.mp4", 1:length(ds["time"]), framerate=30) do n
    @info "Animating double gyre b profiles frame $n/$(length(ds["time"]))..."
    frame[] = n
end

close(ds)
