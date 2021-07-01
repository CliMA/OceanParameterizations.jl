using JLD2

RESULTS_DIR = "C:\\Users\\xinle\\Documents\\OceanParameterizations.jl\\final_results"
NDE_DIR = "18sim_old"

function open_NDE_profile(filepath)
    file = jldopen(filepath)
    NDE_profile = file["NDE_profile"]
    close(file)
    return NDE_profile
end

files_training = [
    "train_wind_-5e-4_cooling_3e-8_new"   
    "train_wind_-5e-4_cooling_2e-8_new"   
    "train_wind_-5e-4_cooling_1e-8_new"   

    "train_wind_-3.5e-4_cooling_3e-8_new" 
    "train_wind_-3.5e-4_cooling_2e-8_new" 
    "train_wind_-3.5e-4_cooling_1e-8_new" 

    "train_wind_-2e-4_cooling_3e-8_new"   
    "train_wind_-2e-4_cooling_2e-8_new"   
    "train_wind_-2e-4_cooling_1e-8_new"   

    "train_wind_-5e-4_heating_-3e-8_new"  
    "train_wind_-5e-4_heating_-2e-8_new"  
    "train_wind_-5e-4_heating_-1e-8_new"  

    "train_wind_-3.5e-4_heating_-3e-8_new"
    "train_wind_-3.5e-4_heating_-2e-8_new"
    "train_wind_-3.5e-4_heating_-1e-8_new"

    "train_wind_-2e-4_heating_-3e-8_new"  
    "train_wind_-2e-4_heating_-2e-8_new"  
    "train_wind_-2e-4_heating_-1e-8_new"  
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

function calculate_losses(NDE_profiles)
    @inline T_loss(data) = sum(data) / 1153

    return [
        (
            NN = T_loss(profile["T_losses"]),
            mpp = T_loss(profile["T_losses_modified_pacanowski_philander"]),
            kpp = T_loss(profile["T_losses_kpp"]),
        ) for profile in NDE_profiles
    ]
end

@show calculate_losses(NDE_profiles_training)


files_training = [
    "test_wind_-4.5e-4_cooling_2.5e-8" 
    "test_wind_-4.5e-4_cooling_1.5e-8" 
    "test_wind_-2.5e-4_cooling_2.5e-8" 
    "test_wind_-2.5e-4_cooling_1.5e-8" 

    "test_wind_-4.5e-4_heating_-2.5e-8"
    "test_wind_-4.5e-4_heating_-1.5e-8"
    "test_wind_-2.5e-4_heating_-2.5e-8"
    "test_wind_-2.5e-4_heating_-1.5e-8"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)



files_training = [
    "test_wind_-5.5e-4_new"
    "test_wind_-5.5e-4_cooling_3.5e-8"
    "test_wind_-5.5e-4_heating_-3.5e-8"

    "test_wind_-1.5e-4_cooling_3.5e-8"
    "test_wind_-1.5e-4_heating_-3.5e-8"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)


NDE_DIR = "3sim_diurnal"

files_training = [
    "test_wind_-5.5e-4_new"
    "test_wind_-5.5e-4_cooling_3.5e-8"
    "test_wind_-5.5e-4_heating_-3.5e-8"

    "test_wind_-1.5e-4_cooling_3.5e-8"
    "test_wind_-1.5e-4_heating_-3.5e-8"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)

NDE_DIR = "18sim_old"

files_training = [
    "test_wind_-5.5e-4_diurnal_5.5e-8"
    "test_wind_-1.5e-4_diurnal_5.5e-8"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)

NDE_DIR = "3sim_diurnal"

files_training = [
    "test_wind_-5.5e-4_diurnal_5.5e-8"
    "test_wind_-1.5e-4_diurnal_5.5e-8"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)

files_training = [
    "test_wind_-4.5e-4_diurnal_4e-8"
    "test_wind_-4.5e-4_diurnal_2e-8"
    "test_wind_-3e-4_diurnal_4e-8"  
    "test_wind_-3e-4_diurnal_2e-8"  
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)

files_training = [
    "train_wind_-5e-4_diurnal_5e-8"  
    "train_wind_-3.5e-4_diurnal_5e-8"  
    "train_wind_-2e-4_diurnal_5e-8"  
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

@show calculate_losses(NDE_profiles_training)