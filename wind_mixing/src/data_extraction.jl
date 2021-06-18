function extract_NN(FILE_PATH, OUTPUT_PATH, type)
    @info "Opening file"
    if type == "NDE"
        train_files, train_parameters, losses, loss_scalings, uw_NN, vw_NN, wT_NN, optimizer = jldopen(FILE_PATH, "r") do file
            train_files = file["training_info/train_files"]
            N_stages = length(keys(file["training_data/neural_network/uw"]))
            @show N_stages
            N_data = length(keys(file["training_data/neural_network/uw/$N_stages"]))

            if "parameters" in keys(file["training_info"])
                train_parameters = file["training_info/parameters"]
            else
                train_parameters = nothing
            end

            total_losses = Array{Float32}(undef, N_data)

            if "loss_scalings" in keys(file["training_info"])
                loss_scalings = file["training_info/loss_scalings"]
                profile_losses = similar(total_losses)
                gradient_losses = similar(total_losses)
                u_losses = similar(total_losses)
                v_losses = similar(total_losses)
                T_losses = similar(total_losses)
                ∂u∂z_losses = similar(total_losses)
                ∂v∂z_losses = similar(total_losses)
                ∂T∂z_losses = similar(total_losses)

                @info "Loading Losses"

                for i in 1:length(total_losses)
                    total_losses[i] = file["training_data/loss/total/$N_stages/$i"]
                    profile_losses[i] = file["training_data/loss/profile/$N_stages/$i"]
                    gradient_losses[i] = file["training_data/loss/gradient/$N_stages/$i"]
                    u_losses[i] = file["training_data/loss/u/$N_stages/$i"]
                    v_losses[i] = file["training_data/loss/v/$N_stages/$i"]
                    T_losses[i] = file["training_data/loss/T/$N_stages/$i"]
                    ∂u∂z_losses[i] = file["training_data/loss/∂u∂z/$N_stages/$i"]
                    ∂v∂z_losses[i] = file["training_data/loss/∂v∂z/$N_stages/$i"]
                    ∂T∂z_losses[i] = file["training_data/loss/∂T∂z/$N_stages/$i"]
                end

            else
                loss_scalings = nothing
                @info "Loading Total Loss"
                for i in 1:length(total_losses)
                    total_losses[i] = file["training_data/loss/$N_stages/$i"]
                    profile_losses = nothing
                    gradient_losses = nothing
                    u_losses = nothing
                    v_losses = nothing
                    T_losses = nothing
                    ∂u∂z_losses = nothing
                    ∂v∂z_losses = nothing
                    ∂T∂z_losses = nothing      
                end
            end

            losses = (
                total = total_losses,
              profile = profile_losses,
             gradient = gradient_losses,
                    u = u_losses,
                    v = v_losses,
                    T = T_losses,
                 ∂u∂z = ∂u∂z_losses,
                 ∂v∂z = ∂v∂z_losses,
                 ∂T∂z = ∂T∂z_losses
             )

            @info "Loading NN"
            NN_index = argmin(total_losses)
            uw_NN = file["training_data/neural_network/uw/$(N_stages)/$NN_index"]
            vw_NN = file["training_data/neural_network/vw/$(N_stages)/$NN_index"]
            wT_NN = file["training_data/neural_network/wT/$(N_stages)/$NN_index"]

            
            if "optimizer" in keys(file["training_data"])
                @info "Loading Optimizer State"
                η = file["training_data/optimizer/η/$(N_stages)/$NN_index"]
                β = file["training_data/optimizer/β/$(N_stages)/$NN_index"]
                state = file["training_data/optimizer/state/$(N_stages)/$NN_index"]
                optimizer = (; η, β, state)
            else
                optimizer = nothing
            end
            return train_files, train_parameters, losses, loss_scalings, uw_NN, vw_NN, wT_NN, optimizer
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
            file["losses/total"] = losses.total
            file["losses/profile"] = losses.profile
            file["losses/gradient"] = losses.gradient
            file["losses/u"] = losses.u
            file["losses/v"] = losses.v
            file["losses/T"] = losses.T
            file["losses/∂u∂z"] = losses.∂u∂z
            file["losses/∂v∂z"] = losses.∂v∂z
            file["losses/∂T∂z"] = losses.∂T∂z

            file["neural_network/uw"] = uw_NN
            file["neural_network/vw"] = vw_NN
            file["neural_network/wT"] = wT_NN

            file["optimizer/η"] = optimizer.η
            file["optimizer/β"] = optimizer.β
            file["optimizer/state"] = optimizer.state
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
   
        train_parameters = Dict()
        train_parameters["train_epochs"] = file["training_info/train_epochs"]
        train_parameters["train_tranges"] = file["training_info/train_tranges"]
        train_parameters["parameters"] = file["training_info/parameters"]

        N_data = length(keys(file["training_data/loss/total"]))

        total_losses = Array{Float32}(undef, N_data)
        
        u_losses = similar(total_losses)
        v_losses = similar(total_losses)
        T_losses = similar(total_losses)

        ∂u∂z_losses = similar(total_losses)
        ∂v∂z_losses = similar(total_losses)
        ∂T∂z_losses = similar(total_losses)

        @info "Loading Loss"
        for i in 1:length(total_losses)
            total_losses[i] = file["training_data/loss/total/$i"]

            u_losses[i] = file["training_data/loss/u/$i"]
            v_losses[i] = file["training_data/loss/v/$i"]
            T_losses[i] = file["training_data/loss/T/$i"]
            
            ∂u∂z_losses[i] = file["training_data/loss/∂u∂z/$i"]
            ∂v∂z_losses[i] = file["training_data/loss/∂v∂z/$i"]
            ∂T∂z_losses[i] = file["training_data/loss/∂T∂z/$i"]
        end

        profile_losses = u_losses .+ v_losses .+ T_losses
        gradient_losses = ∂u∂z_losses .+ ∂v∂z_losses .+ ∂T∂z_losses

        losses = (
            total = total_losses,
            profile = profile_losses,
            gradient = gradient_losses,
            u = u_losses,
            v = v_losses,
            T = T_losses,
            ∂u∂z = ∂u∂z_losses,
            ∂v∂z = ∂v∂z_losses,
            ∂T∂z = ∂T∂z_losses,
        )


        @info "Loading Best Parameters"
        NN_index = argmin(total_losses)
        parameters = file["training_data/parameters/$NN_index"]
        return train_files, train_parameters, losses, parameters
    end

    @info "Writing file"
    jldopen(OUTPUT_PATH, "w") do file
        @info "Writing Training Info"
        file["training_info/train_files"] = train_files
        file["training_info/parameters"] = train_parameters


        @info "Writing Loss"
        
        file["losses/total"] = losses.total
        file["losses/profile"] = losses.profile
        file["losses/gradient"] = losses.gradient

        file["losses/u"] = losses.u
        file["losses/v"] = losses.v
        file["losses/T"] = losses.T

        file["losses/∂u∂z"] = losses.∂u∂z
        file["losses/∂v∂z"] = losses.∂v∂z
        file["losses/∂T∂z"] = losses.∂T∂z

        @info "Writing Best Parameters"
        file["parameters"] = parameters
    end

    @info "End"
end