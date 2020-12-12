"""
    animate_variable(v, v̄; xlabel, filepath, frameskip=1, fps=15)

Create an animation of the variable `v` and coarse-grained variable `v̄`.
`xlabel` should be specified for the plot. A `filepath should also be specified`.
A `.gif` and `.mp4` will be saved. `frameskip` > 1 can be used to skip some frames
to produce the animation more quickly. The output animation will use `fps` frames
per second.
"""
function animate_variable(v, v̄; xlabel, filepath, frameskip=1, fps=15)
    times = dims(v, Ti)
    z = dims(v, ZDim)
    z̄ = dims(v̄, ZDim)

    xlims = extrema(v)
    ylims = (minimum(z), 0)

    kwargs = (linewidth=3, linealpha=0.8, xlims=xlims, ylims=ylims, xlabel=xlabel,
              ylabel="Depth z (meters)", grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    anim = @animate for n=1:frameskip:length(times)
        @info "Plotting $(v.name) for $filepath [$n/$(length(times))]..."

        time_str = @sprintf("%.2f days", times[n] / days)

        plot(v[Ti=n][:], z[:], label="LES (Nz=$(length(z)))", title="Free convection: $time_str"; kwargs...)
        plot!(v̄[Ti=n][:], z̄[:], label="coarse (Nz=$(length(z̄)))"; kwargs...)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end

function animate_learned_heat_flux(ds, NN, T_scaling, wT_scaling; filepath, frameskip=1, fps=15)
    Nz, Nt = size(ds[:T])
    times = dims(ds[:wT], Ti)
    zf = dims(ds[:wT], ZDim)

    xlims = extrema(ds[:wT])
    ylims = (minimum(zf), 0)

    kwargs = (linewidth=3, linealpha=0.8, xlims=xlims, ylims=ylims, xlabel="Heat flux wT (m/s °C)",
              ylabel="Depth z (meters)", grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $filepath [$n/$Nt]..."

        time_str = @sprintf("%.2f days", times[n] / days)

        plot(ds[:wT][Ti=n][:], zf[:], label="Oceananigans", title="Free convection: $time_str"; kwargs...)

        temperature = T_scaling.(ds[:T][Ti=n].data)
        bottom_flux = wT_scaling(ds[:wT][Ti=n, zF=1])
        top_flux = wT_scaling(ds[:wT][Ti=n, zF=Nz+1])
        input = FreeConvectionTrainingDataInput(temperature, bottom_flux, top_flux)
        wT_NN = unscale.(NN(input), Ref(wT_scaling))

        plot!(wT_NN, zf[:], label="neural network"; kwargs...)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end