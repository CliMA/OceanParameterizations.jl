function predict_NN(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

function save_NN_weights(weights, FILE_PATH, filename)
    NN_params = Dict(:weights => weights)
    bson(joinpath(FILE_PATH, "$filename.bson"), NN_params)
end

function train_NN(NN, uvT, flux, optimizers, train_epochs, FILE_PATH, NN_type)
    function prepare_training_data(input, truth)
        return [(input[:,i], truth[:,i]) for i in 1:size(truth, 2)]
    end

    train_data = shuffle(prepare_training_data(uvT, flux))

    loss(x, y) = Flux.Losses.mse(predict_NN(NN, x, y), y)

    for i in 1:length(optimizers), epoch in 1:train_epochs[i]
        opt = optimizers[i]
        function cb()
            @info "$NN_type NN, loss = $(mean([loss(train_data[i][1], train_data[i][2]) for i in 1:length(train_data)])), opt $i/$(length(optimizers)), epoch $epoch/$(train_epochs[i])"
        end
        Flux.train!(loss, Flux.params(NN), train_data, opt, cb=Flux.throttle(cb,10))
        write_data_NN_training(FILE_PATH, mean([loss(train_data[i][1], train_data[i][2]) for i in 1:length(train_data)]), NN)
    end

    return Flux.destructure(NN)[1]
end

