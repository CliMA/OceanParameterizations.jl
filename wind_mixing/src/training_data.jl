"""
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, opts, uw_NN, vw_NN, wT_NN)
"""
function write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, opts, uw_NN, vw_NN, wT_NN)
    jldopen(FILE_PATH, "w") do file
        training_info = JLD2.Group(file, "training_info")
        training_info["train_files"] = train_files
        training_info["train_epochs"] = train_epochs
        training_info["train_tranges"] = train_tranges
        training_info["optimizers"] = opts
        training_info["uw_neural_network"] = uw_NN
        training_info["vw_neural_network"] = vw_NN
        training_info["wT_neural_network"] = wT_NN

        training_data = JLD2.Group(file, "training_data")
        loss = JLD2.Group(training_data, "loss")
        neural_network = JLD2.Group(training_data, "neural_network")
        uw = JLD2.Group(neural_network, "uw")
        vw = JLD2.Group(neural_network, "vw")
        wT = JLD2.Group(neural_network, "wT")
    end
end

function write_data_NDE_training(FILE_PATH, loss, uw_NN, vw_NN, wT_NN, stage)
    jldopen(FILE_PATH, "a") do file
        if !haskey(file, "training_data/loss/$stage")
            file["training_data/loss/$stage/1"] = loss
        else
            count = length(keys(file["training_data/loss/$stage"])) + 1
            file["training_data/loss/$stage/$count"] = loss
        end

        if !haskey(file, "training_data/neural_network/uw/$stage")
            file["training_data/neural_network/uw/$stage/1"] = uw_NN
        else
            count = length(keys(file["training_data/neural_network/uw/$stage"])) + 1
            file["training_data/neural_network/uw/$stage/$count"] = uw_NN
        end

        if !haskey(file, "training_data/neural_network/vw/$stage")
            file["training_data/neural_network/vw/$stage/1"] = vw_NN
        else
            count = length(keys(file["training_data/neural_network/vw/$stage"])) + 1
            file["training_data/neural_network/vw/$stage/$count"] = vw_NN
        end

        if !haskey(file, "training_data/neural_network/wT/$stage")
            file["training_data/neural_network/wT/$stage/1"] = wT_NN
        else
            count = length(keys(file["training_data/neural_network/wT/$stage"])) + 1
            file["training_data/neural_network/wT/$stage/$count"] = wT_NN
        end
    end
end

function write_metadata_NN_training(FILE_PATH, train_files, train_epochs, opts, uw_NN, vw_NN, wT_NN)
    jldopen(FILE_PATH, "w") do file
        training_info = JLD2.Group(file, "training_info")
        training_info["train_files"] = train_files
        training_info["train_epochs"] = train_epochs
        training_info["optimizers"] = opts
        training_info["uw_neural_network"] = uw_NN
        training_info["vw_neural_network"] = vw_NN
        training_info["wT_neural_network"] = wT_NN
    end
end

function write_data_NN_training(FILE_PATH, loss, uw_NN, vw_NN, wT_NN)
    # jldopen(FILE_PATH, "a") do file
    #     file["training_data/loss/$stage/$epoch"] = loss
    #     file["training_data/neural_network/uw/$stage/$epoch"] = uw_NN
    #     file["training_data/neural_network/vw/$stage/$epoch"] = vw_NN
    #     file["training_data/neural_network/wT/$stage/$epoch"] = wT_NN
    # end
    
    jldopen(FILE_PATH, "a") do file
        if !haskey(file, "training_data/loss/$stage")
            file["training_data/loss/1"] = loss
        else
            count = length(keys(file["training_data/loss/$stage"])) + 1
            file["training_data/loss/$count"] = loss
        end

        if !haskey(file, "training_data/neural_network/uw/$stage")
            file["training_data/neural_network/uw/1"] = uw_NN
        else
            count = length(keys(file["training_data/neural_network/uw/$stage"])) + 1
            file["training_data/neural_network/uw/$count"] = uw_NN
        end

        if !haskey(file, "training_data/neural_network/vw/$stage")
            file["training_data/neural_network/vw/1"] = vw_NN
        else
            count = length(keys(file["training_data/neural_network/vw/$stage"])) + 1
            file["training_data/neural_network/vw/$count"] = vw_NN
        end

        if !haskey(file, "training_data/neural_network/wT/$stage")
            file["training_data/neural_network/wT/1"] = wT_NN
        else
            count = length(keys(file["training_data/neural_network/wT/$stage"])) + 1
            file["training_data/neural_network/wT/$count"] = wT_NN
        end
    end
end