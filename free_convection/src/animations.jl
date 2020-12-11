"""
    animate_variable(ds, var, loc; grid_points, xlabel, xlim, filepath, frameskip=1, fps=15)

Create an animation of the variable `var` located in the NetCDF dataset `ds`. The coarse-grained version with `grid_points` will also be plotted. `xlabel` and `xlim` should be specified for the plot. A `filepath should also be specified`. `frameskip` > 1 can be used to skip some profiles to produce the animation more quickly. The output animation will be an mp4 with `fps` frames per second.
"""
function animate_variable(ds, var; grid_points, xlabel, xlim, filepath, frameskip=1, fps=15)
    Nz, Nt = size(ds[var])
    loc = location_z(ds[var])

    if loc == Cell
        z_fine = ds["zC"]
        z_coarse = coarse_grain(ds["zC"], grid_points, Cell)
    elseif loc == Face
        z_fine = ds["zF"]
        z_coarse = coarse_grain(ds["zF"], grid_points+1, Face)
    end

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $var for $filepath [$n/$Nt]..."
        var_fine = ds[var][:, n]

        if loc == Cell
            var_coarse = coarse_grain(ds[var][:, n], grid_points, Cell)
        elseif loc == Face
            var_coarse = coarse_grain(ds[var][:, n], grid_points+1, Face)
        end

        time_str = @sprintf("%.2f days", ds["time"][n] / days)

        plot(var_fine, z_fine, linewidth=2, xlim=xlim, ylim=(-100, 0),
             label="fine (Nz=$(length(z_fine)))", xlabel=xlabel, ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(var_coarse, z_coarse, linewidth=2, label="coarse (Nz=$(length(z_coarse)))")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)
    gif(anim, filepath*".gif", fps=fps)

    return nothing
end
