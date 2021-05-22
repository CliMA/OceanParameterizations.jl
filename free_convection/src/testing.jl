function compute_nde_solution_history(datasets, NDEType, algorithm, nn_filepath, nn_history_filepath)
    final_nn = jldopen(nn_filepath, "r")

    Nz = final_nn["grid_points"]
    NN = final_nn["neural_network"]
    T_scaling = final_nn["T_scaling"]
    wT_scaling = final_nn["wT_scaling"]

    history = jldopen(nn_history_filepath, "r")
    epochs = keys(history["neural_network"]) |> length

    ids = keys(datasets)
    solution_history = Dict(id => [] for id in ids)

    for e in 1:epochs
        @info "Computing NDE solutions for epoch $e/$epochs..."

        NN = history["neural_network/$e"]

        for (id, ds) in datasets
            nde_sol = solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling)
            push!(solution_history[id], nde_sol)
        end

        e % 50 == 0 && GC.gc() # Mercy if you're running this on a little laptop.
    end

    close(final_nn)
    close(history)

    return solution_history
end

function plot_epoch_loss(ids_train, ids_test, nde_solutions, true_solutions, T_scaling; title, filepath)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])

    p = plot(dpi=200)

    ylims = (1e-1, 1e-1)
    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, ylabel="Mean squared error",
              xlabel="Epochs", title=title, grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    for id in ids
        true_sol_scaled = T_scaling.(true_solutions[id].T)
        loss_history = [Flux.mse(true_sol_scaled, T_scaling.(nde_solutions[id][e].T)) for e in 1:epochs]

        min_loss, max_loss = extrema(loss_history)
        ylims_min = min(ylims[1], 10^floor(log10(min_loss)))
        ylims_max = max(ylims[2], 10^ceil(log10(max_loss)))
        ylims = (ylims_min, ylims_max)
        label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
        linestyle = id in ids_train ? :solid : :dot

        plot!(p, 1:epochs, loss_history, label=label, linestyle=linestyle, ylims=ylims, xlims=(1, epochs); kwargs...)
    end

    savefig(filepath)

    return p
end

function animate_nde_loss(datasets, ids_train, ids_test, nde_solutions, true_solutions, T_scaling; title, filepath, fps=15)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])
    times = dims(datasets[first(ids)][:T], Ti)[:] ./ days

    ylims=(1e-6, 1)
    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, xlabel="Simulation time (days)",
              ylabel="Mean squared error", grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    anim = @animate for e in 1:epochs
        @info "Plotting NDE loss evolution... epoch $e/$epochs"
        title_epoch = title * ": epoch $e"

        p = plot(dpi=200)

        for id in ids
            true_sol_scaled = T_scaling.(true_solutions[id].T)
            nde_sol_scaled = T_scaling.(nde_solutions[id][e].T)
            nde_loss = Flux.mse(true_sol_scaled, nde_sol_scaled, agg=x->mean(x, dims=1))[:]

            min_loss, max_loss = extrema(filter(!iszero, nde_loss))
            ylims = (min(ylims[1], 10^floor(log10(min_loss))), max(ylims[2], 10^ceil(log10(max_loss))))

            label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
            linestyle = id in ids_train ? :solid : :dot

            plot!(p, times, nde_loss, label=label, title=title_epoch, linestyle=linestyle,
                  xlims=extrema(times), ylims=ylims; kwargs...)
        end
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end

minimum_nonzero(xs...) = min([minimum(filter(!iszero, x)) for x in xs]...)
maximum_nonzero(xs...) = max([maximum(filter(!iszero, x)) for x in xs]...)

function plot_comparisons(ds, id, ids_train, nde_sol, kpp_sol, tke_sol, convective_adjustment_sol, oceananigans_sol, T_scaling; filepath, frameskip=1, fps=15)
    Nz, Nt = size(ds[:T])
    zc = dims(ds[:T], ZDim) |> Array
    zf = dims(ds[:wT], ZDim) |> Array
    times = dims(ds[:wT], Ti) ./ days

    kwargs = (linewidth=2, linealpha=0.8, grid=false, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    T_lims = extrema(ds[:T])
    wT_lims = extrema(ds[:wT])

    loss(T, T̂) = Flux.mse(T_scaling.(T), T_scaling.(T̂))
    loss_nde = [loss(ds[:T][Ti=n][:], nde_sol.T[:, n]) for n in 1:Nt]
    loss_kpp = [loss(ds[:T][Ti=n][:], kpp_sol.T[:, n]) for n in 1:Nt]
    loss_tke = [loss(ds[:T][Ti=n][:], tke_sol.T[:, n]) for n in 1:Nt]
    loss_ca = [loss(ds[:T][Ti=n][:], convective_adjustment_sol.T[:, n]) for n in 1:Nt]
    loss_emb = [loss(ds[:T][Ti=n][:], oceananigans_sol.T[:, n]) for n in 1:Nt]

    loss_min = minimum_nonzero(loss_nde, loss_kpp, loss_tke, loss_ca, loss_emb)
    loss_max = maximum_nonzero(loss_nde, loss_kpp, loss_tke, loss_ca, loss_emb)
    loss_extrema = (loss_min, loss_max)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting comparisons [frame $n/$Nt]: $filepath ..."

        time_str = @sprintf("%.2f days", times[n])
        title = @sprintf("Free convection (Q = %d W/m², %s): %s", -ds.metadata[:heat_flux_Wm⁻²], id in ids_train ? "train" : "test", time_str)

        wT_plot = plot(margin=5Plots.mm)
        T_plot = plot(margin=5Plots.mm)

        plot!(wT_plot, ds[:wT][Ti=n][:], zf, xlims=wT_lims, label="LES", color="dodgerblue", ylims=extrema(zf),
              xlabel="Heat flux w′T′ (m/s ⋅ K)", ylabel="Depth z (meters)", legend=nothing; kwargs...)

        plot!(T_plot, ds[:T][Ti=n][:], zc, label="LES", color="dodgerblue", xlabel="Temperature T (°C)",
              xlims=T_lims, ylims=extrema(zf), title=title, legend=:bottomright; kwargs...)

        # plot!(T_plot, convective_adjustment_sol.T[:, n], zc, label="Convective adjustment", color="gray"; kwargs...)

        plot!(wT_plot, nde_sol.wT[:, n], zf, label="NDE", color="forestgreen"; kwargs...)
        plot!(T_plot, nde_sol.T[:, n], zc, label="NDE", color="forestgreen"; kwargs...)

        # plot!(wT_plot, oceananigans_sol.wT[:, n], zf, label="Embedded", color="darkorange", linestyle=:dot; kwargs...)
        # plot!(T_plot, oceananigans_sol.T[:, n], zc, label="Embedded", color="darkorange", linestyle=:dot; kwargs...)

        # plot!(wT_plot, kpp_sol.wT[:, n], zf, label="KPP", color="crimson"; kwargs...)
        # plot!(T_plot, kpp_sol.T[:, n], zc, label="KPP", color="crimson"; kwargs...)

        # plot!(wT_plot, tke_sol.wT[:, n], zf, label="TKE mass flux", color="darkmagenta"; kwargs...)
        # plot!(T_plot, tke_sol.T[:, n], zc, label="TKE mass flux", color="darkmagenta"; kwargs...)

        loss_plot = plot(margin=5Plots.mm)

        time_in_days = times[1:n]

        plot!(loss_plot, time_in_days, loss_nde[1:n], label="NDE", color="forestgreen",
              yaxis=(:log10, (1e-6, 1e-1)), xlims=extrema(times), ylims=(1e-6, 1e-1),
              xlabel="Time (days)", ylabel="Mean squared error", legend=nothing; kwargs...)

        plot!(loss_plot, time_in_days, loss_nde[1:n], label="NDE", color="forestgreen"; kwargs...)
        # plot!(loss_plot, time_in_days, loss_emb[1:n], label="Embedded", color="darkorange", linestyle=:dot; kwargs...)
        # plot!(loss_plot, time_in_days, loss_kpp[1:n], label="KPP", color="crimson"; kwargs...)
        # plot!(loss_plot, time_in_days, loss_tke[1:n], label="TKE mass flux", color="darkmagenta"; kwargs...)

        plot(wT_plot, T_plot, loss_plot, layout=(1, 3), size=(1000, 400), dpi=200)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)
end

function plot_loss_matrix(datasets, ids_train, nde_sols, kpp_sols, tke_sols, convective_adjustment_sols, oceananigans_sols, T_scaling; filepath)

    loss(T, T̂) = Flux.mse(T_scaling.(T), T_scaling.(T̂))

    plots = []

    for (id, ds) in datasets
        Nz, Nt = size(ds[:T])
        times = dims(ds[:wT], Ti) ./ days

        kwargs = (linewidth=2, linealpha=0.8, grid=false, framestyle=:box, yaxis=(:log10, (1e-6, 1e-1)),
                  xlabel="Time (days)", ylabel="Mean squared error", xlims=extrema(times), ylims=(1e-6, 1e-1),
                  title = @sprintf("Q = %d W/m² (%s)", -ds.metadata[:heat_flux_Wm⁻²], id in ids_train ? "train" : "test"),
                  legend=isempty(plots) ? :bottomright : nothing,
                  foreground_color_legend=nothing, background_color_legend=nothing)

        loss_nde = [loss(ds[:T][Ti=n][:], nde_sols[id].T[:, n]) for n in 1:Nt]
        loss_kpp = [loss(ds[:T][Ti=n][:], kpp_sols[id].T[:, n]) for n in 1:Nt]
        loss_tke = [loss(ds[:T][Ti=n][:], tke_sols[id].T[:, n]) for n in 1:Nt]
        loss_ca = [loss(ds[:T][Ti=n][:], convective_adjustment_sols[id].T[:, n]) for n in 1:Nt]
        loss_emb = [loss(ds[:T][Ti=n][:], oceananigans_sols[id].T[:, n]) for n in 1:Nt]

        p = plot()

        # plot!(p, times, loss_ca, label="Convective adjustment", color="gray"; kwargs...)
        plot!(p, times, loss_nde, label="NDE", color="forestgreen"; kwargs...)
        # plot!(p, times, loss_emb, label="Embedded", color="darkorange", linestyle=:dot; kwargs...)
        # plot!(p, times, loss_kpp, label="KPP", color="crimson"; kwargs...)
        # plot!(p, times, loss_tke, label="TKE mass flux", color="darkmagenta"; kwargs...)

        push!(plots, p)
    end

    P = plot(plots..., size=(800, 600), dpi=200)

    savefig(filepath)

    return P
end
