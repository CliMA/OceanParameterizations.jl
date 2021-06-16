using WindMixing

DATA_NAME = "NDE_training_mpp_18sim_windcooling_windheating_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_2e-4"
EXTRACTED_DATA_DIR = joinpath(pwd(), "extracted_training_output")
OUTPUT_DIR = "D:\\Output_o"

test_files = [
    # "wind_-5e-4_cooling_4e-8", 
    # "wind_-1e-3_cooling_4e-8", 
    # "wind_-2e-4_cooling_1e-8", 
    # "wind_-1e-3_cooling_2e-8", 
    # "wind_-5e-4_cooling_1e-8", 
    # "wind_-2e-4_cooling_5e-8", 
    # "wind_-5e-4_cooling_3e-8", 
    "wind_-2e-4_cooling_3e-8", 
    # "wind_-1e-3_cooling_3e-8", 
    # "wind_-1e-3_heating_-4e-8",
    # "wind_-1e-3_heating_-1e-8",
    # "wind_-1e-3_heating_-3e-8",
    # "wind_-5e-4_heating_-5e-8",
    # "wind_-5e-4_heating_-3e-8",
    # "wind_-5e-4_heating_-1e-8",
    # "wind_-2e-4_heating_-5e-8",
    # "wind_-2e-4_heating_-3e-8",
    # "wind_-2e-4_heating_-1e-8",
]

# animate_training_results(test_files, DATA_NAME, trange=1:1:1153)

animate_training_results(test_files, DATA_NAME,
                         EXTRACTED_DATA_DIR=EXTRACTED_DATA_DIR, OUTPUT_DIR=OUTPUT_DIR)