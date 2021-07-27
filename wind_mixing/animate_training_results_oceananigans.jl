using WindMixing

FILE_NAME = "NDE_18sim_windcooling_windheating_18simBFGST0.8grad_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8"
EXTRACTED_DATA_DIR = joinpath(pwd(), "extracted_training_output/")

# test_files = [
#     "wind_-3.5e-4_diurnal_3.5e-8"
# ]

test_files = [
    "wind_-5e-4_cooling_3e-8_new",   
    # "wind_-5e-4_cooling_2e-8_new",   
    # "wind_-5e-4_cooling_1e-8_new",   
    # "wind_-3.5e-4_cooling_3e-8_new", 
    # "wind_-3.5e-4_cooling_2e-8_new", 
    # "wind_-3.5e-4_cooling_1e-8_new", 
    # "wind_-2e-4_cooling_3e-8_new",   
    # "wind_-2e-4_cooling_2e-8_new",   
    # "wind_-2e-4_cooling_1e-8_new",   
    # "wind_-5e-4_heating_-3e-8_new",  
    # "wind_-5e-4_heating_-2e-8_new",  
    # "wind_-5e-4_heating_-1e-8_new",  
    # "wind_-3.5e-4_heating_-3e-8_new",
    # "wind_-3.5e-4_heating_-2e-8_new",
    # "wind_-3.5e-4_heating_-1e-8_new",
    # "wind_-2e-4_heating_-3e-8_new",  
    # "wind_-2e-4_heating_-2e-8_new",  
    # "wind_-2e-4_heating_-1e-8_new",
]

OCEANANIGANS_OUTPUT_DIR = joinpath("D:\\Output_o\\CA")
# solve_oceananigans_modified_pacanowski_philander_nn(test_files, "extracted_training_output/$(FILE_NAME)_extracted.jld2", OCEANANIGANS_OUTPUT_DIR, timestep=600)

animate_training_results(test_files, FILE_NAME,
                            EXTRACTED_DATA_DIR=EXTRACTED_DATA_DIR, OUTPUT_DIR=OCEANANIGANS_OUTPUT_DIR, convective_adjustment=true)
