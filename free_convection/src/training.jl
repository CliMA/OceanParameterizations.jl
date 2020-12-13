function inscribe_history(history_filepath, NN, loss)
    jldopen(history_filepath, "a") do file
        if !haskey(file, "loss")
            file["loss/1"] = loss
        else
            epochs = length(keys(file["loss"])) + 1
            file["loss/$epochs"] = loss
        end

        if !haskey(file, "neural_network")
            file["neural_network/1"] = NN
        else
            epochs = length(keys(file["neural_network"])) + 1
            file["neural_network/$epochs"] = NN
        end
    end
    return nothing
end

inscribe_history(::Nothing, args...) = nothing

function train_neural_differential_equation!(NN, training_datasets, T_scaling, wT_scaling, iterations, opt, epochs; history_filepath=nothing)

    ids = [id for id in keys(training_datasets)] |> sort

    nde_params = Dict(id => FreeConvectionNDEParameters(training_datasets[id], T_scaling, wT_scaling) for id in ids)
    T₀ = Dict(id => T_scaling.(training_datasets[id][:T][Ti=1].data) for id in ids)
    ndes = Dict(id => FreeConvectionNDE(NN, training_datasets[id]; iterations) for id in ids)

    true_sols = [T_scaling.(training_datasets[id][:T][Ti=iterations].data) for id in ids]
    true_sols = cat(true_sols..., dims=2)

    function nde_loss()
        nde_sols = [solve_free_convection_nde(ndes[id], NN, T₀[id], Tsit5(), nde_params[id]) |> Array
                    for id in ids]
        nde_sols = cat(nde_sols..., dims=2)
        return Flux.mse(nde_sols, true_sols)
    end

    # @info "Benchmarking loss function..."

    function nde_callback()
        mse_loss = nde_loss()
        @info @sprintf("Training free convection NDE... MSE loss = %.12e", mse_loss)
        inscribe_history(history_filepath, NN, mse_loss)
        return nothing
    end

    Flux.train!(nde_loss, Flux.params(NN), Iterators.repeated((), epochs), opt, cb=nde_callback)

    return nothing
end