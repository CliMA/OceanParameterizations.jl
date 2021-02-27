using JLD2
using FileIO
using Flux
using OrdinaryDiffEq, DiffEqSensitivity
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--input"
        help = "location of input file in training_output directory"
        required = true
    "--output"
        help = "destination of output file in extracted_training_output directory"
        required = true
    "--type"
        help = "training type: NDE or NN"
        default = "NDE"
end

arg_parse = parse_args(s)

FILE_PATH = joinpath(pwd(), "training_output", arg_parse["input"])
OUTPUT_PATH = joinpath(pwd(), "extracted_training_output", arg_parse["output"])
# OUTPUT_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", arg_parse["output"])
type = arg_parse["type"]

@info "Opening file"
if type == "NDE"
    train_files, losses, uw_NN, vw_NN, wT_NN = jldopen(FILE_PATH, "r") do file
        train_files = file["training_info/train_files"]
        N_stages = length(keys(file["training_data/loss"]))
        N_data = length(keys(file["training_data/loss/$N_stages"]))
        losses = Array{Float32}(undef, N_data)
        
        @info "Loading Loss"
        for i in 1:length(losses)
            losses[i] = file["training_data/loss/$N_stages/$i"]
        end

        @info "Loading NN"
        NN_index = argmin(losses)
        uw_NN = file["training_data/neural_network/uw/$(N_stages)/$NN_index"]
        vw_NN = file["training_data/neural_network/vw/$(N_stages)/$NN_index"]
        wT_NN = file["training_data/neural_network/wT/$(N_stages)/$NN_index"]
        return train_files, losses, uw_NN, vw_NN, wT_NN
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