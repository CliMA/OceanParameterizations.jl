using WindMixing

DATA_NAMES = [
              # "NDE_training_mpp_4sim_wind_mixing_-1e-3_-5e-4_cooling_5e-8_2e-8_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4",
              # "NDE_training_mpp_4sim_wind_mixing_-1e-3_-5e-4_cooling_5e-8_2e-8_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_2e-4",
              # "NDE_training_mpp_4sim_wind_mixing_-1e-3_-5e-4_cooling_5e-8_2e-8_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4",
              # "NDE_training_mpp_4sim_wind_mixing_-1e-3_-5e-4_cooling_5e-8_2e-8_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4",
              "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4_5e-5",
              "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4_2e-5",
              "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_2e-4_5e-5",
              "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4_2e-5",
                ]

# test_files = ["-1e-3", "-5e-4", "cooling_5e-8", "cooling_2e-8", "-7e-4", "-3e-4", "cooling_3e-8", "cooling_1e-8", "wind_-5e-4_cooling_3e-8", "wind_-1e-3_cooling_2e-8"]
test_files = ["-1e-3", "-9e-4", "-8e-4", "-6e-4", "-5e-4", "-7e-4", "-3e-4", "-4e-4", "-2e-4"]

for DATA_NAME in DATA_NAMES
  animate_training_results(test_files, DATA_NAME, trange=1:1:1153)
end