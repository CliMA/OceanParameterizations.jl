directories = Dict(
    "free_convection"          => "2DaySuite/three_layer_constant_fluxes_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection_statistics.jld2",
    "strong_wind"              => "2DaySuite/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2",
    "strong_wind_no_coriolis"  => "2DaySuite/three_layer_constant_fluxes_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation_statistics.jld2",
    "weak_wind_strong_cooling" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu3.0e-04_Qb1.0e-07_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling_statistics.jld2",
    "strong_wind_weak_cooling" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu8.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling_statistics.jld2",
    "strong_wind_weak_heating" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb-4.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_heating_statistics.jld2",
    "-1e-3"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-9e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu9.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-8e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu8.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-7e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu7.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-6e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu6.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-5e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-4e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-3e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "-2e-4"                          => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2",
    "cooling_6e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb6.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "cooling_5e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "cooling_4e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb4.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "cooling_3e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb3.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "cooling_2e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb2.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "cooling_1e-8"                   => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb1.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "heating_-3e-8"                  => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2",
    "wind_-5e-4_cooling_4e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-1e-3_cooling_4e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb4.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-2e-4_cooling_1e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-1e-3_cooling_2e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb2.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-5e-4_cooling_1e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-2e-4_cooling_5e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-5e-4_cooling_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-2e-4_cooling_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-1e-3_cooling_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-1e-3_heating_-4e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb-4.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-1e-3_heating_-1e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-1e-3_heating_-3e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-5e-4_heating_-5e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-5.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-5e-4_heating_-3e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-5e-4_heating_-1e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-2e-4_heating_-5e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-5.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-2e-4_heating_-3e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-2e-4_heating_-1e-8"       => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    
    #anything above this line is old and should not be used
    "wind_-5e-4_cooling_3e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-5e-4_cooling_2e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-5e-4_cooling_1e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-3.5e-4_cooling_3e-8_new"  => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-3.5e-4_cooling_2e-8_new"  => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-3.5e-4_cooling_1e-8_new"  => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-2e-4_cooling_3e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-2e-4_cooling_2e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-2e-4_cooling_1e-8_new"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_WC_8days_new_statistics.jld2",
    "wind_-5e-4_heating_-3e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-5e-4_heating_-2e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-2.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-5e-4_heating_-1e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-3.5e-4_heating_-3e-8_new" => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-3.5e-4_heating_-2e-8_new" => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb-2.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-3.5e-4_heating_-1e-8_new" => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-2e-4_heating_-3e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-3.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-2e-4_heating_-2e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-2.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",
    "wind_-2e-4_heating_-1e-8_new"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb-1.0e-08_f1.0e-04_Nh256_Nz128_WH_8days_new_statistics.jld2",

    "wind_-5e-4_diurnal_5e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-5e-4_diurnal_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-5e-4_diurnal_1e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    
    "wind_-3.5e-4_diurnal_5e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3.5e-4_diurnal_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3.5e-4_diurnal_1e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    
    "wind_-2e-4_diurnal_5e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-2e-4_diurnal_3e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-2e-4_diurnal_1e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb1.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    
    "wind_-2e-4_diurnal_2e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-2e-4_diurnal_3.5e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb3.5e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3.5e-4_diurnal_2e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3.5e-4_diurnal_3.5e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb3.5e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-5e-4_diurnal_2e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-5e-4_diurnal_3.5e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.5e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",

    "cooling_5e-8_new"             => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_4.5e-8_new"           => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb4.5e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_4e-8_new"             => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb4.0e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_3.5e-8_new"           => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb3.5e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_3e-8_new"             => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb3.0e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_2.5e-8_new"           => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb2.5e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_2e-8_new"             => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb2.0e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_1.5e-8_new"           => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb1.5e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",
    "cooling_1e-8_new"             => "Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb1.0e-08_f1.0e-04_Nh256_Nz128_FC_8days_statistics.jld2",

    "wind_-5e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-4.5e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-4e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-3.5e-4_new"              => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.5e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-3e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-2.5e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.5e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",
    "wind_-2e-4_new"                => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",

    "wind_-4.5e-4_cooling_2.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb2.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-2.5e-4_cooling_1.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.5e-04_Qb1.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-4.5e-4_cooling_1.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb1.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-2.5e-4_cooling_2.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.5e-04_Qb2.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",

    "wind_-4.5e-4_heating_-2.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb-2.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-2.5e-4_heating_-1.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.5e-04_Qb-1.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-4.5e-4_heating_-1.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb-1.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-2.5e-4_heating_-2.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.5e-04_Qb-2.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",

    "wind_-4.5e-4_diurnal_4e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-4.5e-4_diurnal_2e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu4.5e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3e-4_diurnal_4e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.0e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-3e-4_diurnal_2e-8"      => "Data/three_layer_constant_fluxes_linear_hr192_Qu3.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",

    "wind_-2e-4_diurnal_2e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",

    "wind_-5.5e-4_diurnal_5.5e-8"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.5e-04_Qb5.5e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-1.5e-4_diurnal_5.5e-8"    => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.5e-04_Qb5.5e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",
    "wind_-2e-4_diurnal_4e-8"        => "Data/three_layer_constant_fluxes_linear_hr192_Qu2.0e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2",

    "wind_-5.5e-4_new"              => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.5e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_WM_8days_new_statistics.jld2",

    "wind_-5.5e-4_heating_-3.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.5e-04_Qb-3.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",
    "wind_-1.5e-4_heating_-3.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.5e-04_Qb-3.5e-08_f1.0e-04_Nh256_Nz128_WH_8days_statistics.jld2",

    "wind_-5.5e-4_cooling_3.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu5.5e-04_Qb3.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
    "wind_-1.5e-4_cooling_3.5e-8"   => "Data/three_layer_constant_fluxes_linear_hr192_Qu1.5e-04_Qb3.5e-08_f1.0e-04_Nh256_Nz128_WC_8days_statistics.jld2",
)

datadeps_files = Dict(
    "wind_-5e-4_cooling_3e-8" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC/",
    "wind_-5e-4_cooling_3e-8_cubic" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_constant_fluxes_cubic_hr192_Qu5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_WC_cubic/",
    "diurnal_Qu_-5e-4_diurnal_Qb_3e-8" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_sinusoidal_Qu_sinusoidal_Qb_cubic_hr192_Qu-5.0e-042.4e+010.0e+000.0e+00_Qb3.0e-082.4e+010.0e+000.0e+00_f1.0e-04_Nh256_Nz128_diurnal_Qb_Qu/",
    "constant_Qu_-5e-4_constant_Qb_3e-8" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_constant_fluxes_cubic_hr192_Qu-5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_constant_Qb_Qu/",
    "constant_Qu_-5e-4_constant_Qb_3e-8_2" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_constant_fluxes_cubic_hr192_Qu-5.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_constant_Qb_Qu_2/",
    "diurnal_Qu_-5e-4_diurnal_Qb_3e-8_2" => "https://engaging-web.mit.edu/~xinkai/OceanParameterizations.jl/Data/three_layer_sinusoidal_Qu_sinusoidal_Qb_cubic_hr192_Qu-5.0e-042.4e+010.0e+000.0e+00_Qb3.0e-082.4e+010.0e+000.0e+00_f1.0e-04_Nh256_Nz128_diurnal_Qb_Qu_2/",
)

function load_data(filenames; Nz_coarse=32, scaling=ZeroMeanUnitVarianceScaling)
    !isa(filenames, Array) && (filenames = [filenames])

    for name in filenames
        @assert haskey(datadeps_files, name)
    end

    for name in filenames
        register(DataDep(name, "LES generated using Oceananigans.jl", "$(datadeps_files[name])/instantaneous_statistics_with_halos.jld2", sha2_256))
    end

    FILE_PATHS = [@datadep_str "$(name)/instantaneous_statistics_with_halos.jld2" for name in filenames]

    TrainingDatasets(FILE_PATHS, Nz_coarse=Nz_coarse, scaling=scaling)
end

function diurnal_fluxes(train_files, constants)
    α, g = constants.α, constants.g
    output = Array{Function}(undef,length(train_files))

    @inline wT_flux(Qᵇ, t)::Float32 = Qᵇ * sin(2π / (24 * 60 ^ 2) * t) / (α * g)
    
    for (i, train_file) in enumerate(train_files)
        if occursin("diurnal_5.5e-8", train_file)
            Q = 5.5f-8
        elseif occursin("diurnal_5e-8", train_file)
            Q = 5f-8
        elseif occursin("diurnal_4e-8", train_file)
            Q = 4f-8
        elseif occursin("diurnal_3.5e-8", train_file)
            Q = 3.5f-8
        elseif occursin("diurnal_3e-8", train_file)
            Q = 3f-8
        elseif occursin("diurnal_2e-8", train_file)
            Q = 2f-8
        elseif occursin("diurnal_1e-8", train_file)
            Q = 1f-8
        end
        output[i] = t -> wT_flux(Q, t)
    end
    return output
end

function read_les_output(filename::String)
    filename = joinpath(pwd(), directories[filename])
    return ReadJLD2_LESbraryData(filename)
end

"""
# Description
Takes NzxNt arrays of profiles for variables u, v, T and returns
Nzx(Nt-1) arrays of the profile evolution for u, v, T, u'w', v'w', and w'T'
the horizontally averaged flux for variable V.

# Arguments
Unscaled u, v, T, z, t, and f
"""
function reconstruct_flux_profiles(u, v, T, νₑ_∂z_u, νₑ_∂z_v, κₑ_∂z_T, zF, t, f)

    Δz = diff(zF)
    Δt = diff(t, dims=1)'

    Nz, Nt = size(T)

    ∂t(A) = (A[:,2:Nt] .- A[:,1:Nt-1]) ./ Δt
    dudt = ∂t(u) # Nz x (Nt-1) array of approximate dUdt values
    dvdt = ∂t(v) # Nz x (Nt-1) array of approximate dVdt values
    dTdt = ∂t(T) # Nz x (Nt-1) array of approximate dTdt values

    ∂z(A) = (A[1:Nz,:] .- A[2:Nz+1,:]) ./ Δz
    νₑ_∂²z_u = ∂z(νₑ_∂z_u)
    νₑ_∂²z_v = ∂z(νₑ_∂z_v)
    κₑ_∂²z_T = ∂z(κₑ_∂z_T)

    # remove the last timestep from the variables that were not differentiated w.r.t t
    u = u[:,1:Nt-1]
    v = v[:,1:Nt-1]
    T = T[:,1:Nt-1]
    νₑ_∂²z_u = νₑ_∂²z_u[:,1:Nt-1]
    νₑ_∂²z_v = νₑ_∂²z_v[:,1:Nt-1]
    κₑ_∂²z_T = κₑ_∂²z_T[:,1:Nt-1]

    """ evaluates wϕ = ∫ ∂z(wϕ) dz """
    function wϕ(∂z_wϕ)
        ans = zeros(Nz+1, Nt-1) # one fewer column than T
        for i in 1:Nt-1, h in 1:Nz-1
            ans[h+1, i] = ans[h, i] + Δz[h] * ∂z_wϕ[h, i]
        end
        return ans
    end

    # duw_dz = -dudt .+ f*v .+ νₑ_∂²z_u
    # dvw_dz = -dvdt .- f*u .+ νₑ_∂²z_v
    # dwT_dz = -dTdt .+ κₑ_∂²z_T

    # Without subgrid fluxes:
    duw_dz = -dudt .+ f*v
    dvw_dz = -dvdt .- f*u
    dwT_dz = -dTdt

    # u, v, T, uw, vw, wT, t
    return (u, v, T, wϕ(duw_dz), wϕ(dvw_dz), wϕ(dwT_dz), t[1:Nt-1])
end

struct FluxData{Z, C, S, U, T} # for each of uw, vw, and wT
                z :: Z # z vector for the variable
           coarse :: C # Nz x Nt array of unscaled profiles
           scaled :: S # Nz x Nt array of scaled profiles
       unscale_fn :: U # function to unscaled profile vectors with
       uvT_scaled :: S # unsubsampled uvT profiles
    training_data :: T # subsampled (uvT, scaled) pairs
end

struct uvTData{Z, C, S, U} # for each of u, v, T
             z :: Z # z vector for the variable
        coarse :: C # Nz x Nt array of unscaled profiles
        scaled :: S # Nz x Nt array of scaled profiles
    unscale_fn :: U # function to unscaled profile vectors with
end

struct ProfileData{Σ, U, V, Θ, UW, VW, WT, T, D}
    grid_points :: Int
   uvT_unscaled :: Σ  # 3Nz x Nt array
     uvT_scaled :: Σ  # 3Nz x Nt array
              u :: U
              v :: V
              T :: Θ
             uw :: UW
             vw :: VW
             wT :: WT
              t :: T  # timeseries Vector
       scalings :: D  # Dict mapping names (e.g. "uw") to the AbstractFeatureScaling object associated with it.
end

"""
    coarse_grain(field::Field, new_grid; dims=3)
Coarse grain a `field` onto a `new_grid` along `dims`. Returns a new `Field`.
"""
function coarse_grain(field::Field{X, Y, Center}, new_grid; dims=3) where {X, Y}

    # TODO: Generalize to x and y.
    @assert dims == 3
    @assert new_grid.Nx == 1
    @assert new_grid.Ny == 1

    # TODO: Generalize `coarse_grain` to non-integer ratios.
    r = field.grid.Nz / new_grid.Nz
    @assert isinteger(r)
    r = Int(r)

    coarse_field = Field(location(field)..., field.architecture, new_grid, field.boundary_conditions)

    coarse_data = zeros(size(coarse_field))
    field_interior = interior(field)[1, 1, :]
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz, Center)

    set!(coarse_field, coarse_data)

    return coarse_field
end

function coarse_grain(field::Field{X, Y, Face}, new_grid; dims=3) where {X, Y}

    # TODO: Generalize to x and y.
    @assert dims == 3
    @assert new_grid.Nx == 1
    @assert new_grid.Ny == 1

    r = field.grid.Nz / new_grid.Nz

    coarse_field = Field(location(field)..., field.architecture, new_grid, field.boundary_conditions)

    coarse_data = zeros(size(location(coarse_field), new_grid))
    field_interior = interior(field)[1, 1, :]
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz+1, Face)

    set!(coarse_field, coarse_data)

    return coarse_field
end

function coarse_grain(fts::FieldTimeSeries, new_grid)
    fts_new = FieldTimeSeries(new_grid, location(fts), fts.times; ArrayType=Array{Float32})

    Nt = size(fts, 4)
    for n in 1:Nt
        fts_new.data[:, :, :, n] .= coarse_grain(fts[n], new_grid).data
    end

    return fts_new
end

function coarse_grain(fds::FieldDataset, new_grid)
    coarse_fields = Dict{String, FieldTimeSeries}(
        name => coarse_grain(fts, new_grid)
        for (name, fts) in fds.fields
    )
    return FieldDataset(coarse_fields, fds.metadata, fds.filepath)
end

"""
    data(filenames; animate=false, scale_type=MinMaxScaling, animate_dir="Output", override_scalings=nothing, reconstruct_fluxes=false)

# Arguments
- filenames                "free_convection"
- animate                  Whether to save an animation of all the original profiles
- animate_dir              Directory to save the animation files to
- scale_type               ZeroMeanUnitVarianceScaling or MinMaxScaling
- override_scalings::Dict  For if you want the testing simulation data to be scaled in the same way as the training data.
                           Set to 𝒟train.scalings to use the scalings from 𝒟train.
"""
function data(filenames; animate=false, scale_type=MinMaxScaling, animate_dir="Output",
                override_scalings=nothing, reconstruct_fluxes=false, subsample_frequency=1,enforce_surface_fluxes=false)

    filenames isa String && (filenames = [filenames])

    # Harvest data from Oceananigans simulation output files.
    all_les = Dict()

    for file in filenames
        all_les[file] = read_les_output(file)
    end

    get_array(f) = cat((f(les) for (file, les) in all_les)..., dims=2)

    u  = get_array(les -> les.U)
    v  = get_array(les -> les.V)
    T  = get_array(les -> les.T)
    # νₑ_∂z_u = get_array(les -> les.νₑ_∂z_u)
    # νₑ_∂z_v = get_array(les -> les.νₑ_∂z_v)
    # κₑ_∂z_T = get_array(les -> les.κₑ_∂z_T)
    t  = cat((les.t for (file, les) in all_les)..., dims=1)

    function enforce_top_surface_flux!(A, flux)
        A[end,:] .= flux
        return A
    end

    vw = get_array(les -> les.wv)
    if enforce_surface_fluxes
        uw = get_array(les -> enforce_top_surface_flux!(les.wu, les.u_top))
        wT = get_array(les -> enforce_top_surface_flux!(les.wT, les.θ_top))
    else
        uw = get_array(les -> les.wu)
        wT = get_array(les -> les.wT)
    end

    first = all_les[filenames[1]]
    zC = first.zC
    zF = first.zF

    NzC = length(zC)
    NzF = length(zF)
    Nt = sum([length(les.t) for (file, les) in all_les])

    if reconstruct_fluxes
        Nt -= length(all_les)
        u = zeros(NzC, Nt)
        v = zeros(NzC, Nt)
        T = zeros(NzC, Nt)
        uw = zeros(NzF, Nt)
        vw = zeros(NzF, Nt)
        wT = zeros(NzF, Nt)
        t = zeros(Nt)
        x=1
        for (file,les) in all_les
            Nt = length(les.t)-1
            cols = x:x+Nt-1 # time indices corresponding to the current simulation
            # @. u[:,cols], v[:,cols], T[:,cols], uw[:,cols], vw[:,cols], wT[:,cols], t[cols] = reconstruct_flux_profiles(les.U, les.V, les.T, les.νₑ_∂z_u, les.νₑ_∂z_v, les.κₑ_∂z_T, zF, les.t, les.f⁰)
            a = reconstruct_flux_profiles(les.U, les.V, les.T, les.νₑ_∂z_u, les.νₑ_∂z_v, les.κₑ_∂z_T, zF, les.t, les.f⁰)
            u[:,cols]  .= a[1]
            v[:,cols]  .= a[2]
            T[:,cols]  .= a[3]
            uw[:,cols] .= a[4]
            vw[:,cols] .= a[5]
            wT[:,cols] .= a[6]
            t[cols]    .= a[7]
            if enforce_surface_fluxes
                uw[end,cols] .= les.u_top
                wT[end,cols] .= les.θ_top
            end
            x += Nt
        end
    end

    if animate
        animate_gif([uw], zF, t, "uw", directory=animate_dir)
        animate_gif([vw], zF, t, "vw", directory=animate_dir)
        animate_gif([wT], zF, t, "wT", directory=animate_dir)
        animate_gif([u],  zC, t, "u",  directory=animate_dir)
        animate_gif([v],  zC, t, "v",  directory=animate_dir)
        animate_gif([T],  zC, t, "T",  directory=animate_dir)
    end

    function coarsify_cell(x)
        output = zeros(typeof(x[1]), 32, size(x,2))
        for i in 1:size(x,2)
            col = @view output[:,i]
            col .= coarse_grain(x[:,i], 32, Center)
        end
        return output
    end

    function coarsify_face(x)
        output = zeros(typeof(x[1]), 33, size(x,2))
        for i in 1:size(x,2)
            col = @view output[:,i]
            col .= coarse_grain_linear_interpolation(x[:,i], 33, Face)
            # col .= coarse_grain(x[:,i], 33, Face)
        end
        return output
    end
    # coarsify_cell(x) = cat((coarse_grain(x[:,i], 32, Center) for i in 1:size(x,2))..., dims=2)
    # coarsify_face(x) = cat((coarse_grain_linear_interpolation(x[:,i], 33, Face) for i in 1:size(x,2))..., dims=2)
    # coarsify_face(x) = cat((coarse_grain(x[:,i], 33, Face) for i in 1:size(x,2))..., dims=2)

    u_coarse  = coarsify_cell(u)
    v_coarse  = coarsify_cell(v)
    T_coarse  = coarsify_cell(T)
    uw_coarse = coarsify_face(uw)
    vw_coarse = coarsify_face(vw)
    wT_coarse = coarsify_face(wT)
    # νₑ_∂z_u   = coarsify_face(νₑ_∂z_u)
    # νₑ_∂z_v   = coarsify_face(νₑ_∂z_v)
    # κₑ_∂z_T   = coarsify_face(κₑ_∂z_T)

    zC_coarse = coarse_grain(zC, 32, Center)
    zF_coarse = coarse_grain_linear_interpolation(zF, 33, Face)
    # zF_coarse = coarse_grain(zF, 33, Face)

    function get_scaling(name, coarse)
        if isnothing(override_scalings)
            # set the scaling according to the data (for training simulations)
            return scale_type(coarse)
        else
            # for if you want the testing simulation data to be scaled in the same way as the training data
            return override_scalings[name]
        end
    end

    all_scalings=Dict()

    for (name, coarse) in [("u", u_coarse), ("v", v_coarse), ("T", T_coarse),
                           ("uw", uw_coarse), ("vw", vw_coarse), ("wT", wT_coarse)]
        all_scalings[name] = get_scaling(name, coarse)
    end

    training_set = collect(1:subsample_frequency:length(t))

    function get_scaled(name, coarse)
        scaling = get_scaling(name, coarse)
        scaled = all_scalings[name].(coarse)
        unscale_fn = Base.inv(scaling)
        return (scaled, unscale_fn)
    end

    function get_uvTData(name, coarse, z)
        scaled, unscale_fn = get_scaled(name, coarse)
        return uvTData(z, coarse, scaled, unscale_fn)
    end

    function get_FluxData(name, coarse, z)
        scaled, unscale_fn = get_scaled(name, coarse)
        training_data = [(uvT_scaled[:,i], scaled[:,i]) for i in training_set] # (predictor, target) pairs
        return FluxData(z, coarse, scaled, unscale_fn, uvT_scaled, training_data)
    end

    u = get_uvTData("u", u_coarse, zC_coarse)
    v = get_uvTData("v", v_coarse, zC_coarse)
    T = get_uvTData("T", T_coarse, zC_coarse)
    uvT_unscaled = cat(u.coarse, v.coarse, T.coarse, dims=1)
    uvT_scaled = cat(u.scaled, v.scaled, T.scaled, dims=1)

    uw = get_FluxData("uw", uw_coarse, zF_coarse)
    vw = get_FluxData("vw", vw_coarse, zF_coarse)
    wT = get_FluxData("wT", wT_coarse, zF_coarse)

    return ProfileData(33, uvT_unscaled, uvT_scaled, u, v, T, uw, vw, wT, t, all_scalings)
end
