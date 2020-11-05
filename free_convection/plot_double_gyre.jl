using Printf
using NCDatasets
using GeoData
using PyPlot
using PyCall
using Oceananigans.Utils

cmocean = pyimport("cmocean")
const plt = PyPlot
plt.ioff()

ds = NCDstack("double_gyre.nc")

Nx, Ny, Nz, Nt = size(ds[:b])
Nx½, Ny½ = round(Int, Nx/2), round(Int, Ny/2)

xC = ds[:b].dims[1].val / kilometer
xF = ds[:u].dims[1].val / kilometer
yC = ds[:b].dims[2].val / kilometer
yF = ds[:v].dims[2].val / kilometer
zC = ds[:b].dims[3].val
zF = ds[:w].dims[3].val
times = ds[:b].dims[4].val

for n in 1:Nt
    @info "Plotting speed frame $n/$Nt..."
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), dpi=200)
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle("Double gyre, time = $(prettytime(times[n]))", fontsize=16)

    ax_u_surf = axes[1, 1]
    img_u = ax_u_surf.pcolormesh(xF, yC, ds[:speed][zC=Nz, Ti=n]', vmin=0, vmax=1, cmap=cmocean.cm.speed)
    # fig.colorbar(img_u, ax=ax_u_surf, extend="max", label="m/s")
    ax_u_surf.set_xlim(xF[1], xF[end])
    ax_u_surf.set_ylim(yF[1], yF[end])
    ax_u_surf.set_title("speed(x, y, z=0)")
    ax_u_surf.set_xlabel("x (km)")
    ax_u_surf.set_ylabel("y (km)")
    # ax_u_surf.set_aspect("equal")

    ax_u_zonal = axes[1, 2]
    img_u = ax_u_zonal.pcolormesh(xF, zC, ds[:speed][yC=Ny½, Ti=n]', vmin=0, vmax=1, cmap=cmocean.cm.speed)
    # fig.colorbar(img_u, ax=ax_u_zonal, extend="max", label="m/s")
    ax_u_zonal.set_xlim(xF[1], xF[end])
    ax_u_zonal.set_ylim(zF[1], zF[end])
    ax_u_zonal.set_title("speed(x=0, y, z)")
    ax_u_zonal.set_xlabel("x (km)")
    ax_u_zonal.set_ylabel("z (m)")

    ax_u_merid = axes[1, 3]
    img_u = ax_u_merid.pcolormesh(yC, zC, ds[:speed][xF=Nx½, Ti=n]', vmin=0, vmax=1, cmap=cmocean.cm.speed)
    fig.colorbar(img_u, ax=ax_u_merid, extend="max", label="m/s")
    ax_u_merid.set_xlim(yF[1], yF[end])
    ax_u_merid.set_ylim(zF[1], zF[end])
    ax_u_merid.set_title("speed(x, y=0, z)")
    ax_u_merid.set_xlabel("y (km)")
    ax_u_merid.set_ylabel("z (m)")

    plt.savefig(@sprintf("speed_frame_%03d.png", n))
    plt.close(fig)
end
