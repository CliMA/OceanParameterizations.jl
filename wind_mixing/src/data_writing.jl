"""
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, opts, uw_NN, vw_NN, wT_NN)
"""
function write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, opts, uw_NN, vw_NN, wT_NN)
    jldopen(FILE_PATH, "w") do file
        training_info = JLD2.Group(file, "training_info")
        training_info["train_files"] = train_files
        training_info["train_epochs"] = train_epochs
        training_info["train_tranges"] = train_tranges
        training_info["optimizers"] = opts
        training_info["parameters"] = train_parameters
        training_info["uw_neural_network"] = uw_NN
        training_info["vw_neural_network"] = vw_NN
        training_info["wT_neural_network"] = wT_NN

        training_data = JLD2.Group(file, "training_data")
        loss = JLD2.Group(training_data, "loss")
        neural_network = JLD2.Group(training_data, "neural_network")
        uw = JLD2.Group(neural_network, "uw")
        vw = JLD2.Group(neural_network, "vw")
        wT = JLD2.Group(neural_network, "wT")
        η = JLD2.Group(training_data, "η")
        β = JLD2.Group(training_data, "β")
        state = JLD2.Group(training_data, "state")
    end
end

function write_data_NDE_training(FILE_PATH, losses, loss_scalings, uw_NN, vw_NN, wT_NN, stage, optimizer)
    jldopen(FILE_PATH, "a") do file
        profile_loss = losses.u + losses.v + losses.T
        gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z
        total_loss = profile_loss + gradient_loss

        if !haskey(file, "training_data/loss/total/$stage")
            file["training_info/loss_scalings"] = loss_scalings
            file["training_data/loss/total/$stage/1"] = total_loss
            file["training_data/loss/profile/$stage/1"] = profile_loss
            file["training_data/loss/gradient/$stage/1"] = gradient_loss

            file["training_data/loss/u/$stage/1"] = losses.u
            file["training_data/loss/v/$stage/1"] = losses.v
            file["training_data/loss/T/$stage/1"] = losses.T

            file["training_data/loss/∂u∂z/$stage/1"] = losses.∂u∂z
            file["training_data/loss/∂v∂z/$stage/1"] = losses.∂v∂z
            file["training_data/loss/∂T∂z/$stage/1"] = losses.∂T∂z

            file["training_data/neural_network/uw/$stage/1"] = uw_NN
            file["training_data/neural_network/vw/$stage/1"] = vw_NN
            file["training_data/neural_network/wT/$stage/1"] = wT_NN

            file["training_data/optimizer/η/$stage/1"] = optimizer.eta
            file["training_data/optimizer/β/$stage/1"] = optimizer.beta
            file["training_data/optimizer/state/$stage/1"] = optimizer.state
        else
            count = length(keys(file["training_data/loss/total/$stage"])) + 1
            file["training_data/loss/total/$stage/$count"] = total_loss
            file["training_data/loss/profile/$stage/$count"] = profile_loss
            file["training_data/loss/gradient/$stage/$count"] = gradient_loss

            file["training_data/loss/u/$stage/$count"] = losses.u
            file["training_data/loss/v/$stage/$count"] = losses.v
            file["training_data/loss/T/$stage/$count"] = losses.T

            file["training_data/loss/∂u∂z/$stage/$count"] = losses.∂u∂z
            file["training_data/loss/∂v∂z/$stage/$count"] = losses.∂v∂z
            file["training_data/loss/∂T∂z/$stage/$count"] = losses.∂T∂z

            file["training_data/neural_network/uw/$stage/$count"] = uw_NN
            file["training_data/neural_network/vw/$stage/$count"] = vw_NN
            file["training_data/neural_network/wT/$stage/$count"] = wT_NN

            file["training_data/optimizer/η/$stage/$count"] = optimizer.eta
            file["training_data/optimizer/β/$stage/$count"] = optimizer.beta
            file["training_data/optimizer/state/$stage/$count"] = optimizer.state
        end
    end
end

function write_metadata_NN_training(FILE_PATH, train_files, train_epochs, opts, NN, NN_type)
    jldopen(FILE_PATH, "w") do file
        training_info = JLD2.Group(file, "training_info")
        training_info["train_files"] = train_files
        training_info["train_epochs"] = train_epochs
        training_info["optimizers"] = opts
        training_info["$(NN_type)_neural_network"] = NN
        training_data = JLD2.Group(file, "training_data")
    end
end

function write_data_NN_training(FILE_PATH, loss, NN)
    jldopen(FILE_PATH, "a") do file
        if !haskey(file, "training_data/loss")
            file["training_data/loss/1"] = loss
        else
            count = length(keys(file["training_data/loss"])) + 1
            file["training_data/loss/$count"] = loss
        end

        if !haskey(file, "training_data/neural_network")
            file["training_data/neural_network/1"] = NN
        else
            count = length(keys(file["training_data/neural_network"])) + 1
            file["training_data/neural_network/$count"] = NN
        end

    end
end

function write_data_NN(FILE_PATH, uw_NN, vw_NN, wT_NN)
    jldopen(FILE_PATH, "w") do file
        file["neural_network/uw"] = uw_NN
        file["neural_network/vw"] = vw_NN
        file["neural_network/wT"] = wT_NN
    end
end

function write_metadata_modified_pacanowski_philander_optimisation(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, opts)
    jldopen(FILE_PATH, "w") do file
        training_info = JLD2.Group(file, "training_info")
        training_info["train_files"] = train_files
        training_info["train_epochs"] = train_epochs
        training_info["train_tranges"] = train_tranges
        training_info["optimizers"] = opts
        training_info["parameters"] = train_parameters
        training_info["loss_scalings"] = loss_scalings

        training_data = JLD2.Group(file, "training_data")
        loss = JLD2.Group(training_data, "loss")
        parameters = JLD2.Group(training_data, "parameters")
    end
end

function write_data_modified_pacanowski_philander_optimisation(FILE_PATH, losses, parameters)
    profile_loss = losses.u + losses.v + losses.T
    gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z
    total_loss = profile_loss + gradient_loss

    jldopen(FILE_PATH, "a") do file
        if !haskey(file, "training_data/loss/total")
            file["training_data/loss/total/1"] = total_loss
            file["training_data/loss/profile/1"] = profile_loss
            file["training_data/loss/gradient/1"] = gradient_loss

            file["training_data/loss/u/1"] = losses.u
            file["training_data/loss/v/1"] = losses.v
            file["training_data/loss/T/1"] = losses.T

            file["training_data/loss/∂u∂z/1"] = losses.∂u∂z
            file["training_data/loss/∂v∂z/1"] = losses.∂v∂z
            file["training_data/loss/∂T∂z/1"] = losses.∂T∂z
            
            file["training_data/parameters/1"] = parameters
        else
            count = length(keys(file["training_data/loss/total"])) + 1

            file["training_data/loss/total/$count"] = total_loss
            file["training_data/loss/profile/$count"] = profile_loss
            file["training_data/loss/gradient/$count"] = gradient_loss

            file["training_data/loss/u/$count"] = losses.u
            file["training_data/loss/v/$count"] = losses.v
            file["training_data/loss/T/$count"] = losses.T

            file["training_data/loss/∂u∂z/$count"] = losses.∂u∂z
            file["training_data/loss/∂v∂z/$count"] = losses.∂v∂z
            file["training_data/loss/∂T∂z/$count"] = losses.∂T∂z

            file["training_data/parameters/$count"] = parameters
        end
    end
end