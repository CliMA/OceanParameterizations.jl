function compute_nde_solution_history(datasets, nn_filepath, algorithm, nn_history_filepath)
    final_nn = jldopen(nn_filepath, "r")
    Nz = final_nn["grid_points"]
    NN = final_nn["neural_network"]
    T_scaling = final_nn["T_scaling"]
    wT_scaling = final_nn["wT_scaling"]

    history = jldopen(nn_history_filepath, "r")
    epochs = keys(history["neural_network"]) |> length

    ids = keys(datasets)
    nde_params = Dict(id => FreeConvectionNDEParameters(datasets[id], T_scaling, wT_scaling) for id in ids)
    T₀ = Dict(id => T_scaling.(datasets[id][:T][Ti=1].data) for id in ids)

    solution_history = Dict(id => [] for id in ids)
    for e in 1:epochs
        @info "Computing NDE solutions for epoch $e/$epochs..."

        NN = history["neural_network/$e"]
        ndes = Dict(id => FreeConvectionNDE(NN, datasets[id]) for id in ids)

        for id in ids
            nde_sol = solve_nde(ndes[id], NN, T₀[id], algorithm, nde_params[id])
            push!(solution_history[id], nde_sol)
        end

        e % 50 == 0 && GC.gc() # Mercy if you're running this on a little laptop.
    end

    close(final_nn)
    close(history)

    return solution_history
end

function plot_epoch_loss(ids_train, ids_test, nde_solutions, true_solutions; title, filepath)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])

    p = plot(dpi=200)

    ylims = (1, 1)
    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, ylabel="Mean squared error",
              xlabel="Epochs", title=title, grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    for id in ids
        loss_history = [Flux.mse(true_solutions[id], nde_solutions[id][e]) for e in 1:epochs]

        min_loss, max_loss = extrema(loss_history)
        ylims = (min(ylims[1], 10^floor(log10(min_loss))), max(ylims[2], 10^ceil(log10(max_loss))))
        label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
        linestyle = id in ids_train ? :solid : :dash

        plot!(p, 1:epochs, loss_history, label=label, linestyle=linestyle, ylims=ylims; kwargs...)
    end

    savefig(filepath)

    return p
end

function animate_nde_loss(datasets, ids_train, ids_test, nde_solutions, true_solutions; title, filepath, fps=15)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])
    times = dims(datasets[first(ids)][:T], Ti)[:] ./ days

    ylims=(1, 1)
    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, xlabel="Simulation time (days)",
              ylabel="Mean squared error", grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    anim = @animate for e in 1:epochs
        @info "Plotting NDE loss evolution... epoch $e/$epochs"
        title_epoch = title * ": epoch $e"

        p = plot(dpi=200)

        for id in ids
            nde_loss = Flux.mse(true_solutions[id], nde_solutions[id][e], agg=x->mean(x, dims=1))[:]

            min_loss, max_loss = extrema(filter(!iszero, nde_loss))
            ylims = (min(ylims[1], 10^floor(log10(min_loss))), max(ylims[2], 10^ceil(log10(max_loss))))

            label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
            linestyle = id in ids_train ? :solid : :dash

            plot!(p, times, nde_loss, label=label, title=title_epoch, linestyle=linestyle, ylims=ylims; kwargs...)
        end
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end

function plot_comparisons(ds, nde_sol, kpp_sol, convective_adjustment_sol, oceananigans_sol; filepath, frameskip=1, fps=15)
    Nz, Nt = size(ds[:T])
    zc = dims(ds[:T], ZDim) |> Array
    zf = dims(ds[:wT], ZDim) |> Array
    times = dims(ds[:wT], Ti)

    kwargs = (linewidth=3, linealpha=0.8, ylims=extrema(zf),
              grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    T_lims = extrema(ds[:T])
    wT_lims = extrema(ds[:wT])

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting comparisons [frame $n/$Nt]: $filepath ..."

        time_str = @sprintf("%.2f days", times[n] / days)

        wT_plot = plot()
        T_plot = plot()

        plot!(wT_plot, ds[:wT][Ti=n][:], zf, label="", xlabel="Heat flux wT (m/s °C)",
              xlims=wT_lims, ylabel="Depth z (meters)", title="Free convection: $time_str"; kwargs...)

        plot!(T_plot, ds[:T][Ti=n][:], zc, label="LES", xlabel="Temperature T (°C)",
              xlims=T_lims; kwargs...)

        plot!(T_plot, convective_adjustment_sol[:, n], zc, label="Convective adjustment"; kwargs...)
        plot!(T_plot, nde_sol[:, n], zc, label="Neural DE"; kwargs...)
        plot!(T_plot, kpp_sol[:, n], zc, label="KPP"; kwargs...)
        plot!(T_plot, oceananigans_sol[:, n], zc, label="Embedded"; kwargs...)

        plot(wT_plot, T_plot, dpi=200)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)
end
