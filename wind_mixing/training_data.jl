using JLD2
using Flux
using FileIO

# OUTPUT_PATH = joinpath(pwd(), "Output")
OUTPUT_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output")


train_files = ["-7.5e-4", "-2.5e-4"]
train_epochs = [100 for i in 1:2]
train_tranges = [1:5:10, 1:5:100]
opts = [[ADAM(), Descent(0.01)], [ADAM()]]
nums = [1, 2]

N_inputs = 96
hidden_units = 400
N_outputs = 31
uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))

FILE_PATH = joinpath(OUTPUT_PATH, "example.jld2")

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
    end
end

write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, opts, uw_NN, vw_NN, wT_NN)

function write_data_NDE_training(FILE_PATH, loss, uw_NN, vw_NN, wT_NN, stage, epoch)
    jldopen(FILE_PATH, "a") do file
        file["training_data/loss/$stage/$epoch"] = loss
        file["training_data/neural_network/uw/$stage/$epoch"] = uw_NN
        file["training_data/neural_network/vw/$stage/$epoch"] = vw_NN
        file["training_data/neural_network/wT/$stage/$epoch"] = wT_NN
    end
end

for i in 1:length(train_epochs), j in 1:train_epochs[i]
    write_data_NDE_training(FILE_PATH, 1e-3, uw_NN, vw_NN, wT_NN, i, j)
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

function write_data_NN_training(FILE_PATH, loss, uw_NN, vw_NN, wT_NN, stage, epoch)
    jldopen(FILE_PATH, "a") do file
        file["training_data/loss/$stage/$epoch"] = loss
        file["training_data/neural_network/uw/$stage/$epoch"] = uw_NN
        file["training_data/neural_network/vw/$stage/$epoch"] = vw_NN
        file["training_data/neural_network/wT/$stage/$epoch"] = wT_NN
    end
end