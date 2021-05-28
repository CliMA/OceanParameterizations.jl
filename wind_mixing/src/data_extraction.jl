function extract_NN(FILE_PATH, OUTPUT_PATH, type)
    @info "Opening file"
    if type == "NDE"
        train_files, train_parameters, losses, uw_NN, vw_NN, wT_NN = jldopen(FILE_PATH, "r") do file
            train_files = file["training_info/train_files"]
            N_stages = length(keys(file["training_data/loss"]))
            N_data = length(keys(file["training_data/loss/$N_stages"]))
            losses = Array{Float32}(undef, N_data)
            
            if "parameters" in keys(file["training_info"])
                train_parameters = file["training_info/parameters"]
            else
                train_parameters = nothing
            end

            @info "Loading Loss"
            for i in 1:length(losses)
                losses[i] = file["training_data/loss/$N_stages/$i"]
            end

            @info "Loading NN"
            NN_index = argmin(losses)
            uw_NN = file["training_data/neural_network/uw/$(N_stages)/$NN_index"]
            vw_NN = file["training_data/neural_network/vw/$(N_stages)/$NN_index"]
            wT_NN = file["training_data/neural_network/wT/$(N_stages)/$NN_index"]
            return train_files, train_parameters, losses, uw_NN, vw_NN, wT_NN
        end
    else
        train_files, losses, NN = jldopen(FILE_PATH, "r") do file
            train_files = file["training_info/train_files"]
            N_data = length(keys(file["training_data/loss"]))
            losses = zeros(N_data)
            
            @info "Loading Loss"
            for i in 1:length(losses)
                losses[i] = file["training_data/loss/$i"]
            end

            @info "Loading NN"
            NN_index = argmin(losses)
            NN = file["training_data/neural_network/$NN_index"]
            return train_files, losses, NN
        end
    end

    @info "Writing file"
    if type == "NDE"
        jldopen(OUTPUT_PATH, "w") do file
            @info "Writing Training Info"
            file["training_info/train_files"] = train_files
            file["training_info/parameters"] = train_parameters


            @info "Writing Loss"
            file["losses"] = losses

            @info "Writing NN"
            file["neural_network/uw"] = uw_NN
            file["neural_network/vw"] = vw_NN
            file["neural_network/wT"] = wT_NN
        end
    else
        jldopen(OUTPUT_PATH, "w") do file
            @info "Writing Training Info"
            file["training_info/train_files"] = train_files

            @info "Writing Loss"
            file["losses"] = losses

            @info "Writing NN"
            file["neural_network"] = NN
        end
    end

    @info "End"
end

function extract_parameters_modified_pacanowski_philander_optimisation(FILE_PATH, OUTPUT_PATH)
    @info "Opening file"
    train_files, train_parameters, losses, parameters = jldopen(FILE_PATH, "r") do file
        train_files = file["training_info/train_files"]
        N_data = length(keys(file["training_data/loss"]))
        losses = Array{Float32}(undef, N_data)
        
        train_parameters = Dict()
        train_parameters["train_epochs"] = file["training_info/train_epochs"]
        train_parameters["train_tranges"] = file["training_info/train_tranges"]
        train_parameters["parameters"] = file["training_info/parameters"]

        @info "Loading Loss"
        for i in 1:length(losses)
            losses[i] = file["training_data/loss/$i"]
        end

        @info "Loading Best Parameters"
        NN_index = argmin(losses)
        parameters = file["training_data/parameters/$NN_index"]
        return train_files, train_parameters, losses, parameters
    end

    @info "Writing file"
    jldopen(OUTPUT_PATH, "w") do file
        @info "Writing Training Info"
        file["training_info/train_files"] = train_files
        file["training_info/parameters"] = train_parameters


        @info "Writing Loss"
        file["losses"] = losses

        @info "Writing Best Parameters"
        file["parameters"] = parameters
    end

    @info "End"
end