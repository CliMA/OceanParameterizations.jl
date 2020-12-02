module Main

using ClimateParameterizations.Data
using ClimateParameterizations.GaussianProcess
using ClimateParameterizations.NeuralNetwork

model_output(x, time_index, ℳ, 𝒟) = GaussianProcess.model_output(𝒟.modify_predictor_fn(x, time_index), ℳ)

predict(𝒱::FluxData, model) = (cat((𝒱.unscale_fn(model(𝒱.training_data[i][1])) for i in 1:length(𝒱.training_data))...,dims=2), 𝒱.coarse)
export predict

# plot hyperparameter landscapes for analysis / optimization
include("hyperparameter_landscapes.jl")
export  plot_landscapes_compare_error_metrics,
        plot_landscapes_compare_files_me,
        plot_error_histogram,
        get_min_gamma,
        get_min_gamma_alpha,
        train_validate_test

include("mean_square_error.jl")
export mean_square_error

end #module
