using WindMixing

DATA_NAMES = [
  "NDE_18sim_windcooling_windheating_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8"
]

EXTRACTED_DATA_DIR = joinpath(pwd(), "extracted_training_output")
OUTPUT_DIR = joinpath("C:\\Users\\xinle\\Documents\\OceanParameterizations.jl\\test_bottom")

test_files = [
  # "wind_-5e-4_diurnal_5e-8"    
  # "wind_-5e-4_diurnal_3e-8"    
  # "wind_-5e-4_diurnal_1e-8"    
      
  # "wind_-3.5e-4_diurnal_5e-8"  
  # "wind_-3.5e-4_diurnal_3e-8"  
  # "wind_-3.5e-4_diurnal_1e-8"  
      
  # "wind_-2e-4_diurnal_5e-8"    
  # "wind_-2e-4_diurnal_3e-8"    
  # "wind_-2e-4_diurnal_1e-8"    
      
  # "wind_-2e-4_diurnal_2e-8"    
  # "wind_-2e-4_diurnal_3.5e-8"  
  # "wind_-3.5e-4_diurnal_2e-8"  
  # "wind_-3.5e-4_diurnal_3.5e-8"
  # "wind_-5e-4_diurnal_2e-8"    
  # "wind_-5e-4_diurnal_3.5e-8"  

  "wind_-5e-4_cooling_3e-8_new",   
  # "wind_-5e-4_cooling_1e-8_new",   
  # "wind_-2e-4_cooling_3e-8_new",   
  # "wind_-2e-4_cooling_1e-8_new",   
  # "wind_-5e-4_heating_-3e-8_new",  
  # "wind_-2e-4_heating_-1e-8_new",  
  # "wind_-2e-4_heating_-3e-8_new",  
  # "wind_-5e-4_heating_-1e-8_new",  

  # "wind_-3.5e-4_cooling_2e-8_new", 
  # "wind_-3.5e-4_heating_-2e-8_new",

  # "wind_-5e-4_cooling_2e-8_new",   
  # "wind_-3.5e-4_cooling_3e-8_new", 
  # "wind_-3.5e-4_cooling_1e-8_new", 
  # "wind_-2e-4_cooling_2e-8_new",   
  # "wind_-3.5e-4_heating_-3e-8_new",
  # "wind_-3.5e-4_heating_-1e-8_new",
  # "wind_-2e-4_heating_-2e-8_new",  
  # "wind_-5e-4_heating_-2e-8_new",  
  ]

for DATA_NAME in DATA_NAMES
    animate_training_results(test_files, DATA_NAME,
                         EXTRACTED_DATA_DIR=EXTRACTED_DATA_DIR, OUTPUT_DIR=OUTPUT_DIR, convective_adjustment=false)
end