using WindMixing
using OceanParameterizations

PATH = joinpath(pwd(), "extracted_training_output")
FILE_PATH = joinpath(pwd(), "Output")

train_files = [
            #    "wind_-5e-4_cooling_4e-8",
               "wind_-1e-3_cooling_4e-8",
               "wind_-2e-4_cooling_1e-8",
               "wind_-1e-3_cooling_2e-8",
            #    "wind_-5e-4_cooling_1e-8",
               "wind_-2e-4_cooling_5e-8",
               "wind_-5e-4_cooling_3e-8",
            #    "wind_-2e-4_cooling_3e-8",
            #    "wind_-1e-3_cooling_3e-8"
               ]

# train_files = ["wind_-1e-3_cooling_4e-8",
#                "wind_-2e-4_cooling_1e-8"]
# train_files = ["wind_-2e-4_cooling_1e-8",]
𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

VIDEO_NAME = "LES_simulations_windcooling"

animate_training_data_profiles_fluxes(train_files, joinpath(FILE_PATH, VIDEO_NAME))
               
               
               
               
               
               
