module Main

using ClimateParameterizations.DataWrangling
using ClimateParameterizations.GaussianProcesses
using ClimateParameterizations.NeuralNetworks

model_output(x, time_index, ℳ, 𝒟) = GaussianProcess.model_output(𝒟.modify_predictor_fn(x, time_index), ℳ)

predict(𝒱::FluxData, model) = (cat((𝒱.unscale_fn(model(𝒱.training_data[i][1])) for i in 1:length(𝒱.training_data))...,dims=2), 𝒱.coarse)
export predict

include("mean_square_error.jl")
export mean_square_error

end #module
