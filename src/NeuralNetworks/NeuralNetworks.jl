module NeuralNetworks

export nn_model

using Statistics
using Flux
using OceanParameterizations.DataWrangling

function cb(train,loss)
    f() = @info mean([loss(train[i][1], train[i][2]) for i in 1:length(train)])
    return f
end

# Return a trained neural network model
function nn_model(; 𝒱=nothing, model=nothing, optimizers=nothing)
    loss(x,y) = Flux.Losses.mse(model(x), y)

    for opt in optimizers
        @info opt
        Flux.train!(loss, params(model), 𝒱.training_data, opt, cb = Flux.throttle(cb(𝒱.training_data,loss), 2))
    end

    return model
end

end #module
