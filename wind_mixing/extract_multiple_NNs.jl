using ArgParse
using WindMixing

type = "NDE"

FILE_NAMES = ["NDE_training_mpp_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4", 
              "NDE_training_mpp_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_2e-4",
              "NDE_training_mpp_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4",
              "NDE_training_mpp_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4"
              ]
FILE_PATHS = [joinpath(pwd(), "training_output", "$(FILE_NAME).jld2") for FILE_NAME in FILE_NAMES]
OUTPUT_PATHS = [joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2") for FILE_NAME in FILE_NAMES]

Threads.@threads for i in 1:length(FILE_NAMES)
    FILE_PATH = FILE_PATHS[i]
    OUTPUT_PATH = OUTPUT_PATHS[i]
    extract_NN(FILE_PATH, OUTPUT_PATH, type)
end

