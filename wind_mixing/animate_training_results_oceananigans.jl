using WindMixing

FILE_NAME = "NDE_training_mpp_10sim_windcooling_windheating_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4"


test_files = [
    "wind_-1e-3_heating_-4e-8",
    "wind_-1e-3_heating_-1e-8",
    "wind_-5e-4_heating_-3e-8",
    "wind_-2e-4_heating_-5e-8",
    "wind_-2e-4_heating_-1e-8",
    "wind_-1e-3_cooling_4e-8",
    "wind_-2e-4_cooling_1e-8",
    "wind_-1e-3_cooling_2e-8",
    "wind_-2e-4_cooling_5e-8",
    "wind_-5e-4_cooling_3e-8",
]

OCEANANIGANS_OUTPUT_DIR = joinpath(pwd(), "Output_o", FILE_NAME)
animate_training_results_oceananigans(test_files, 60, FILE_NAME, OCEANANIGANS_OUTPUT_DIR)