using WindMixing

DATA_NAME = "NDE_training_mpp_8sim_wind_mixing_cooling_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4"

test_files = [
    "-1e-3",       
    "-9e-4",       
    "-8e-4",       
    "-7e-4",       
    "-6e-4",       
    "-5e-4",       
    "-4e-4",       
    "-3e-4",       
    "-2e-4",       
    "cooling_6e-8",
    "cooling_5e-8",
    "cooling_4e-8",
    "cooling_3e-8",
    "cooling_2e-8",
    "cooling_1e-8",
]

test_files = []
animate_training_results(test_files, DATA_NAME, trange=1:1:1153)