include("../les/LESbraryData.jl")

prefix = pwd()*"/src/les/data/"

# time averaged
# directories=Dict(
#     "free_convection"          => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     "strong_wind"              => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     "strong_wind_no_coriolis"  => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_averaged_statistics.jld2",
#     "weak_wind_strong_cooling" => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     "strong_wind_weak_cooling" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     "strong_wind_weak_heating" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
# )

# not time averaged
directories=Dict(
    "free_convection"          => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_statistics.jld2",
    "strong_wind"              => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_statistics.jld2",
    "strong_wind_no_coriolis"  => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_statistics.jld2",
    "weak_wind_strong_cooling" => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_statistics.jld2",
    "strong_wind_weak_cooling" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_statistics.jld2",
    "strong_wind_weak_heating" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_statistics.jld2",
)


"""
read_les_output(filename::String; days=0)
"""
function read_les_output(filename::String)
    filename=prefix*directories[filename]
    return ReadJLD2_LESbraryData(filename)
end

# les = read_les_output("strong_wind_weak_heating")
# les.t

# directories=Dict(
#
#     "free_convection" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_no_coriolis" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "weak_wind_strong_cooling" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_weak_cooling" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
#
#     "strong_wind_weak_heating" => Dict(
#         2 => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2",
#     ),
# )
