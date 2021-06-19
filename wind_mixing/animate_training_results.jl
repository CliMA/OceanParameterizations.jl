using WindMixing

DATA_NAMES = [
    "NDE_2sim_windcooling_SS_windheating_SS_swish_divide1f5_gradient_smallNN_swish_scale_5e-3_rate_2e-4_T0.8_extracted.jld2",
    "NDE_2sim_windcooling_SS_windheating_SS_tanh_divide1f5_gradient_smallNN_tanh_scale_5e-3_rate_2e-4_T0.8_extracted.jld2",
    "NDE_2sim_windcooling_SS_windheating_SS_relu_divide1f5_gradient_smallNN_relu_scale_5e-3_rate_2e-4_T0.5_extracted.jld2",
    "NDE_2sim_windcooling_SS_windheating_SS_relu_divide1f5_gradient_smallNN_relu_scale_5e-3_rate_2e-4_T0.8_extracted.jld2",
    "NDE_2sim_windcooling_SS_windheating_SS_leakyrelu_divide1f5_gradient_smallNN_leakyrelu_scale_5e-3_rate_2e-4_T0.5_extracted.jld2"
]

EXTRACTED_DATA_DIR = joinpath(pwd(), "extracted_training_output")
OUTPUT_DIR = joinpath(pwd(), "Output")

test_files = [
    "wind_-5e-4_cooling_3e-8_new",   
    # "wind_-5e-4_cooling_1e-8_new",   
    # "wind_-2e-4_cooling_3e-8_new",   
    # "wind_-2e-4_cooling_1e-8_new",   
    "wind_-5e-4_heating_-3e-8_new",  
    # "wind_-2e-4_heating_-1e-8_new",  
    # "wind_-2e-4_heating_-3e-8_new",  
    # "wind_-5e-4_heating_-1e-8_new",  
  
    "wind_-3.5e-4_cooling_2e-8_new", 
    "wind_-3.5e-4_heating_-2e-8_new",
  
    # "wind_-5e-4_cooling_2e-8_new",   
    # "wind_-3.5e-4_cooling_3e-8_new", 
    # "wind_-3.5e-4_cooling_1e-8_new", 
    # "wind_-2e-4_cooling_2e-8_new",   
    # "wind_-3.5e-4_heating_-3e-8_new",
    # "wind_-3.5e-4_heating_-1e-8_new",
    # "wind_-2e-4_heating_-2e-8_new",  
    # "wind_-5e-4_heating_-2e-8_new",  
  ]

animate_training_results(test_files, DATA_NAME,
                         EXTRACTED_DATA_DIR=EXTRACTED_DATA_DIR, OUTPUT_DIR=OUTPUT_DIR)