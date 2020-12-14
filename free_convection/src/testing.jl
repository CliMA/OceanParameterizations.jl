function compute_nde_solution_history(datasets, nn_filepath, nn_history_filepath)
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
            nde_sol = solve_nde(ndes[id], NN, T₀[id], Tsit5(), nde_params[id])
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

    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, ylims=(1, 1e4), ylabel="Mean squared error",
              xlabel="Epochs", title=title, grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    for id in ids
        loss_history = [Flux.mse(true_solutions[id], nde_solutions[id][e]) for e in 1:epochs]

        label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
        linestyle = id in ids_train ? :solid : :dash

        plot!(p, 1:epochs, loss_history, label=label, linestyle=linestyle; kwargs...)
    end

    savefig(filepath)

    return p
end

function animate_nde_loss(datasets, ids_train, ids_test, nde_solutions, true_solutions; title, filepath, fps=15)
    ids = (ids_train..., ids_test...)
    epochs = length(nde_solutions[first(ids)])
    times = dims(datasets[first(ids)][:T], Ti)[:] ./ days

    kwargs = (linewidth=2, linealpha=0.8, yaxis=:log, ylims=(1e-1, 1e4), xlabel="Simulation time (days)",
              ylabel="Mean squared error", grid=false, legend=:outertopright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    anim = @animate for e in 1:epochs
        @info "Plotting NDE loss evolution... epoch $e/$epochs"
        title_epoch = title * ": epoch $e"

        p = plot(dpi=200)

        for id in ids
            nde_loss = Flux.mse(true_solutions[id], nde_solutions[id][e], agg=x->mean(x, dims=1))[:]

            label = @sprintf("id=%d (%s)", id, id in ids_train ? "train" : "test")
            linestyle = id in ids_train ? :solid : :dash

            plot!(p, times, nde_loss, label=label, title=title_epoch, linestyle=linestyle; kwargs...)
        end
    end

    @info "Saving $filepath..."
    mp4(anim, filepath * ".mp4", fps=fps)
    gif(anim, filepath * ".gif", fps=fps)

    return nothing
end

