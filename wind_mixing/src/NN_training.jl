using Statistics
using Flux
using OceanParameterizations
using Oceananigans.Grids
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing

function predict_NN(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

function save_NN_weights(weights, FILE_PATH, filename)
    NN_params = Dict(:weights => weights)
    bson(joinpath(FILE_PATH, "$filename.bson"), NN_params)
end

function train_NN(NN, uvT, flux, optimizers, epochs=1, FILE_PATH=pwd(), filename="weights")
    function prepare_training_data(input, truth)
        return [(input[:,i], truth[:,i]) for i in 1:size(truth, 2)]
    end

    data = prepare_training_data(uvT, flux)

    loss(x, y) = Flux.Losses.mse(predict_NN(NN, x, y), y)

    function cb()
        @info "loss = $(mean([loss(data[i][1], data[i][2]) for i in 1:length(data)]))"
    end

    for opt in optimizers, epoch in 1:epochs
        @info "Epoch $epoch/$epochs, $opt"
        Flux.train!(loss, Flux.params(NN), data, opt, cb=Flux.throttle(cb, 5))
        save_NN_weights(Flux.destructure(NN)[1], FILE_PATH, filename)
    end

    return Flux.destructure(NN)[1]
end

