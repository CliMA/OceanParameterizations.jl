import Zygote: Params, pullback

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

function dense_spatial_causality_train!(loss, ps, data, opt; cb = () -> ())
    #= Breaks in Zygote due to mutation...  =#
    local training_loss
    ps = Params(ps)
    for d in data
        gs = gradient(ps) do
            training_loss = loss(d...)
            return training_loss
        end
        for i in 1:2:length(ps)
            nrows, ncols = size(ps[i])
            mask = [x >= y ? 1.0 : 0.0 for x in 1:nrows, y in 1:ncols]
            ps[i] .*= mask
        end
        Flux.update!(opt, ps, gs)
        cb()
    end

end

function train_neural_differential_equation!(NN, NDEType, algorithm, datasets, T_scaling, wT_scaling, iterations, opt, epochs; history_filepath=nothing, causal_penalty=nothing)

    ids = [id for id in keys(datasets)] |> sort

    nde_params = Dict(id => FreeConvectionNDEParameters(datasets[id], T_scaling, wT_scaling) for id in ids)
    T₀ = Dict(id => T_scaling.(interior(datasets[id]["T"])[1, 1, :, 1]) for id in ids)
    ndes = Dict(id => NDEType(NN, datasets[id]; iterations) for id in ids)

    true_sols = [T_scaling.(interior(datasets[id]["T"])[1, 1, :, iterations]) for id in ids]
    true_sols = cat(true_sols..., dims=2)

    function nde_loss()
        nde_sols = cat([solve_nde(ndes[id], NN, T₀[id], algorithm, nde_params[id]) |> Array for id in ids]..., dims=2)
        if !isnothing(causal_penalty)
            return Flux.mse(nde_sols, true_sols) + causal_penalty(NN)
        else
            return Flux.mse(nde_sols, true_sols)
        end
     end

    function nde_callback()
        mse_loss = nde_loss()
        @info @sprintf("Training free convection NDE... MSE loss: μ_loss::%s = %.12e", typeof(mse_loss), mse_loss)
        inscribe_history(history_filepath, NN, mse_loss)
        return nothing
    end

    Flux.train!(nde_loss, Flux.params(NN), Iterators.repeated((), epochs), opt, cb=nde_callback)

    return nothing
end
