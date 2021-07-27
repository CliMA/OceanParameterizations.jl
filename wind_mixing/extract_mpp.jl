FILE_NAME = "parameter_optimisation_18sim_windcooling_windheating_5params_BFGS_T0.8_grad"

FILE_PATH = "training_output/$(FILE_NAME).jld2"
OUTPUT_PATH = "D:\\Output_o\\$(FILE_NAME)_extracted.jld2"

extract_parameters_modified_pacanowski_philander_optimisation(FILE_PATH, OUTPUT_PATH)