using Oceananigans: znodes

"""
    animate_training_data(ds, ds_coarse; filepath, frameskip=1, fps=15)

Create an animation of the variable `v` and coarse-grained variable `v̄`.
`xlabel` should be specified for the plot. A `filepath should also be specified`.
A `.gif` and `.mp4` will be saved. `frameskip` > 1 can be used to skip some frames
to produce the animation more quickly. The output animation will use `fps` frames
per second.
"""
function animate_data(ds, ds_coarse; filepath, frameskip=1, fps=15)

    T = ds["T"]
    wT = ds["wT"]
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times

    T_coarse = ds_coarse["T"]
    wT_coarse = ds_coarse["wT"]
    z̄c = znodes(T_coarse)
    z̄f = znodes(wT_coarse)

    kwargs = (linewidth=3, linealpha=0.8, ylims=extrema(zf),
              grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    T_lims = interior(T) |> extrema
    wT_lims = interior(wT) |> extrema

    anim = @animate for n=1:frameskip:length(times)
        @info "Plotting free convection data [frame $n/$(length(times))]: $filepath..."

        time_str = @sprintf("%.2f days", times[n] / days)

        wT_profile = interior(wT)[1, 1, :, n]
        wT_coarse_profile = interior(wT_coarse)[1, 1, :, n]

        wT_plot = plot()
        plot!(wT_plot, wT_profile, zf, label="LES (Nz=$(length(zf)))", xlabel="Heat flux wT (m/s °C)",
              xlims=wT_lims, ylabel="Depth z (meters)", title="Free convection: $time_str"; kwargs...)
        plot!(wT_plot, wT_coarse_profile, z̄f, label="coarse (Nz=$(length(z̄f)))"; kwargs...)

        T_profile = interior(T)[1, 1, :, n]
        T_coarse_profile = interior(T_coarse)[1, 1, :, n]

        T_plot = plot()
        plot!(T_plot, T_profile, zc, label="LES (Nz=$(length(zc)))", xlabel="Temperature T (°C)",
              xlims=T_lims; kwargs...)
        plot!(T_plot, T_coarse_profile, z̄c, label="coarse (Nz=$(length(z̄c)))"; kwargs...)

        plot(wT_plot, T_plot, dpi=200)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end

function animate_learned_free_convection(ds, NN, NN_function, NDEType, algorithm, T_scaling, wT_scaling; filepath, frameskip=1, fps=15)
    Nz, Nt = size(ds[:T])
    zc = dims(ds[:T], ZDim)
    zf = dims(ds[:wT], ZDim)
    times = dims(ds[:wT], Ti)

    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = T_scaling.(ds[:T][Ti=1].data)
    nde = NDEType(NN, ds)
    nde_sol = solve_nde(nde, NN, T₀, algorithm, nde_params)

    kwargs = (linewidth=3, linealpha=0.8, ylims=extrema(zf),
              grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    T_lims = extrema(ds[:T])
    wT_lims = extrema(ds[:wT])

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting learned free convection [frame $n/$Nt]: $filepath ..."

        time_str = @sprintf("%.2f days", times[n] / days)

        wT_plot = plot()
        T_plot = plot()

        plot!(wT_plot, ds[:wT][Ti=n][:], zf[:], label="Oceananigans", xlabel="Heat flux wT (m/s °C)",
              xlims=wT_lims, ylabel="Depth z (meters)", title="Free convection: $time_str"; kwargs...)

        plot!(T_plot, ds[:T][Ti=n][:], zc[:], label="Oceananigans", xlabel="Temperature T (°C)",
              xlims=T_lims; kwargs...)

        T_NN = nde_sol[n]
        bottom_flux = wT_scaling(ds[:wT][Ti=n, zF=1])
        top_flux = wT_scaling(ds[:wT][Ti=n, zF=Nz+1])
        input = FreeConvectionTrainingDataInput(T_NN, bottom_flux, top_flux)
        wT_NN = unscale.(NN_function(input), Ref(wT_scaling))
        T_NN = unscale.(T_NN, Ref(T_scaling))

        plot!(wT_plot, wT_NN, zf[:], label="neural network"; kwargs...)
        plot!(T_plot, T_NN, zc[:], label="NDE"; kwargs...)

        plot(wT_plot, T_plot, dpi=200)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end
