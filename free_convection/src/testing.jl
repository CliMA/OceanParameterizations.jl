using CairoMakie

function compute_nde_solution_history(datasets, NDEType, algorithm, nn_filepath, nn_history_filepath; gc_interval=50)
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

        e % gc_interval == 0 && GC.gc() # Mercy if you're running this on a little laptop.
    end

    close(final_nn)
    close(history)

    return solution_history
end

function plot_epoch_loss(ids_train, ids_test, nde_solutions, true_solutions, T_scaling; filepath_prefix)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])

    fig = Figure()
    ax1 = fig[1, 1] = Axis(fig, xlabel="Epoch", ylabel="Mean squared error", yscale=log10)

    for id in ids
        true_sol_scaled = T_scaling.(true_solutions[id].T)
        loss_history = [Flux.mse(true_sol_scaled, T_scaling.(nde_solutions[id][e].T)) for e in 1:epochs]
        linestyle = id in ids_train ? :solid : :dot
        lines!(ax1, 1:epochs, loss_history, label="simulation $id"; linestyle)
    end

    CairoMakie.xlims!(0, epochs)
    Legend(fig[1, 2], ax1, framevisible=false)

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

function plot_epoch_loss_summary(ids, nde_solutions, true_solutions, T_scaling; filepath_prefix)
    epochs = length(nde_solutions[first(ids)])

    fig = Figure()
    ax1 = fig[1, 1] = Axis(fig, xlabel="Epoch", ylabel="Mean squared error", yscale=log10)

    function color(id)
        colors = CairoMakie.Makie.wong_colors()
         1 <= id <= 9  && return colors[1]
        10 <= id <= 12 && return colors[2]
        13 <= id <= 15 && return colors[3]
    end

    function label(id)
         1 <= id <= 9  && return "training"
        10 <= id <= 12 && return "Qb interpolation"
        13 <= id <= 15 && return "Qb extrapolation"
    end

    for id in ids
        true_sol_scaled = T_scaling.(true_solutions[id].T)
        loss_history = [Flux.mse(true_sol_scaled, T_scaling.(nde_solutions[id][e].T)) for e in 1:epochs]
        lines!(ax1, 1:epochs, loss_history, color=color(id))
    end

    CairoMakie.xlims!(0, epochs)

    entry1 = LineElement(color=color(1))
    entry2 = LineElement(color=color(10))
    entry3 = LineElement(color=color(13))

    Legend(fig[1, 2], [entry1, entry2, entry3], [label(1), label(10), label(13)], framevisible=false)

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

function animate_nde_loss(datasets, ids_train, ids_test, nde_solutions, true_solutions, T_scaling; title, filepath, fps=15)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])
    times = datasets[first(ids)]["T"].times

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

# TODO: Allow to selectively turn off different parameterizations.
function plot_comparisons(ds, id, ids_train, nde_sol, kpp_sol, tke_sol, convective_adjustment_sol, oceananigans_sol, T_scaling; filepath, frameskip=1, fps=15)
    T = ds["T"]
    wT = ds["wT"]
    FT = eltype(T)
    Nz = size(T, 3)
    Nt = size(T, 4)
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times ./ days

    kwargs = (linewidth=2, linealpha=0.8, grid=false, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    T_lims = interior(T) |> extrema
    wT_lims = interior(wT) |> extrema

    loss(T, T̂) = Flux.mse(T_scaling.(T), T_scaling.(T̂))

    loss_nde = [loss(interior(T)[1, 1, :, n], nde_sol.T[:, n]) for n in 1:Nt]
    loss_kpp = [loss(interior(T)[1, 1, :, n], kpp_sol.T[:, n]) for n in 1:Nt]
    loss_tke = [loss(interior(T)[1, 1, :, n], tke_sol.T[:, n]) for n in 1:Nt]
    loss_ca = [loss(interior(T)[1, 1, :, n], convective_adjustment_sol.T[:, n]) for n in 1:Nt]
    loss_emb = [loss(interior(T)[1, 1, :, n], oceananigans_sol.T[:, n]) for n in 1:Nt]

    loss_min = minimum_nonzero(loss_nde, loss_kpp, loss_tke, loss_ca, loss_emb)
    loss_max = maximum_nonzero(loss_nde, loss_kpp, loss_tke, loss_ca, loss_emb)
    loss_extrema = (loss_min, loss_max)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting comparisons [frame $n/$Nt]: $filepath ..."

        time_str = @sprintf("%.2f days", times[n])
        title = @sprintf("Free convection (Q = %d W/m², %s): %s", -4e6 * ds.metadata["temperature_flux"], id in ids_train ? "train" : "test", time_str)

        wT_plot = plot(margin=5Plots.mm)
        T_plot = plot(margin=5Plots.mm)

        plot!(wT_plot, interior(wT)[1, 1, :, n], zf, xlims=wT_lims, label="LES", color="dodgerblue", ylims=extrema(zf),
              xlabel="Heat flux w′T′ (m/s ⋅ K)", ylabel="Depth z (meters)", legend=nothing; kwargs...)

        plot!(T_plot, interior(T)[1, 1, :, n], zc, label="LES", color="dodgerblue", xlabel="Temperature T (°C)",
              xlims=T_lims, ylims=extrema(zf), title=title, legend=:bottomright; kwargs...)

        plot!(T_plot, convective_adjustment_sol.T[:, n], zc, label="Convective adjustment", color="gray"; kwargs...)

        plot!(wT_plot, nde_sol.wT[:, n], zf, label="NDE", color="forestgreen"; kwargs...)
        plot!(T_plot, nde_sol.T[:, n], zc, label="NDE", color="forestgreen"; kwargs...)

        plot!(wT_plot, oceananigans_sol.wT[:, n], zf, label="Embedded", color="darkorange", linestyle=:dot; kwargs...)
        plot!(T_plot, oceananigans_sol.T[:, n], zc, label="Embedded", color="darkorange", linestyle=:dot; kwargs...)

        plot!(wT_plot, kpp_sol.wT[:, n], zf, label="KPP", color="crimson"; kwargs...)
        plot!(T_plot, kpp_sol.T[:, n], zc, label="KPP", color="crimson"; kwargs...)

        plot!(wT_plot, tke_sol.wT[:, n], zf, label="TKE mass flux", color="darkmagenta"; kwargs...)
        plot!(T_plot, tke_sol.T[:, n], zc, label="TKE mass flux", color="darkmagenta"; kwargs...)

        loss_plot = plot(margin=5Plots.mm)

        time_in_days = times[1:n]

        plot!(loss_plot, time_in_days, loss_nde[1:n], label="NDE", color="forestgreen",
              yaxis=(:log10, (1e-6, 1e-1)), xlims=extrema(times), ylims=(1e-6, 1e-1),
              xlabel="Time (days)", ylabel="Mean squared error", legend=nothing; kwargs...)

        plot!(loss_plot, time_in_days, loss_nde[1:n], label="NDE", color="forestgreen"; kwargs...)
        plot!(loss_plot, time_in_days, loss_emb[1:n], label="Embedded", color="darkorange", linestyle=:dot; kwargs...)
        plot!(loss_plot, time_in_days, loss_kpp[1:n], label="KPP", color="crimson"; kwargs...)
        plot!(loss_plot, time_in_days, loss_tke[1:n], label="TKE mass flux", color="darkmagenta"; kwargs...)

        plot(wT_plot, T_plot, loss_plot, layout=(1, 3), size=(1000, 400), dpi=200)
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)
end

# TODO: Allow to selectively turn off different parameterizations.
function plot_loss_matrix(datasets, ids_train, nde_sols, kpp_sols, tke_sols, convective_adjustment_sols, oceananigans_sols, T_scaling; filepath_prefix,
                          rows = ceil(Int, √length(datasets)),
                          cols = ceil(Int, √length(datasets)),
                          ylims = (1e-6, 1e-1))

    loss(T, T̂) = Flux.mse(T_scaling.(T), T_scaling.(T̂))

    fig = Figure()

    for (d, (id, ds)) in enumerate(datasets)

        T = ds["T"]
        wT = ds["wT"]
        Nz = size(T, 3)
        zc = znodes(T)
        zf = znodes(wT)
        H = abs(zf[1])
        Nt = size(T, 4)
        times = T.times ./ days

        T_solution = [interior(ds["T"])[1, 1, :, n] for n in 1:Nt]
        loss_nde = [loss(T_solution[n], nde_sols[id].T[:, n]) for n in 1:Nt]
        loss_kpp = [loss(T_solution[n], kpp_sols[id].T[:, n]) for n in 1:Nt]
        loss_tke = [loss(T_solution[n], tke_sols[id].T[:, n]) for n in 1:Nt]
        loss_ca = [loss(T_solution[n], convective_adjustment_sols[id].T[:, n]) for n in 1:Nt]
        loss_emb = [loss(T_solution[n], oceananigans_sols[id].T[:, n]) for n in 1:Nt]

        i = div(id-1, rows) + 1
        j = rem(id-1, cols) + 1
        ax_ij = fig[i, j] = Axis(fig, title="simulation $id", xlabel="Time (days)", ylabel="MSE", yscale=log10)

        for loss_i in (loss_ca, loss_nde, loss_emb, loss_kpp, loss_tke)
            replace!(x -> iszero(x) ? NaN : x, loss_i)
        end

        lines!(ax_ij, times, loss_ca,  label="Convective adjustment")
        lines!(ax_ij, times, loss_nde, label="NDE")
        lines!(ax_ij, times, loss_emb, label="NDE (embedded)", linestyle=:dot)
        lines!(ax_ij, times, loss_kpp, label="KPP")
        lines!(ax_ij, times, loss_tke, label="CATKE")

        CairoMakie.xlims!(extrema(times)...)
        CairoMakie.ylims!(ylims...)

        i != rows && hidexdecorations!(ax_ij, grid=false)
        j != 1 && hideydecorations!(ax_ij, grid=false)

        # Add the legend after all axes have been plotted.
        if d == length(datasets)
            Legend(fig[0, :], ax_ij, orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)
        end
    end

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end
