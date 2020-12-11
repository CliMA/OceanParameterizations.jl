using OceanParameterizations
using Makie, MakieLayout

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis", "weak_wind_strong_cooling",
          "strong_wind_weak_cooling", "strong_wind_weak_heating"]

file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_coriolis" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

x_lims = Dict(
    "uw" => (-0.0007,0.0001),
    "vw" => (-0.0001,0.0005),
    "wT" => (-0.00009,0.00006)
)

f = Dict(
    "uw" => 𝒟 -> 𝒟.uw.coarse, "vw" => 𝒟 -> 𝒟.vw.coarse, "wT" => 𝒟 -> 𝒟.wT.coarse, "T"  => 𝒟 -> 𝒟.T.coarse
)

scaling_factor = Dict(
    "uw" => 1e4,
    "vw" => 1e4,
    "wT" => 1e4,
    "T" => 1
)

x_labels = Dict(
    "uw" => "U'W' x 10⁴ (m²/s²)",
    "vw" => "V'W' x 10⁴ (m²/s²)",
    "wT" => "W'T' x 10⁴ (C⋅m/s)",
    "T" => "T (C)"
)

titles = Dict(
    "uw" => "Zonal momentum flux, U'W'",
    "vw" => "Meridional momentum flux, V'W'",
    "wT" => "Temperature flux, W'T'",
    "T" => "Temperature, T",
)

function generate_scene(name)
    scene = Scene()
    for (file, T) in Ts
        z = T.uw.z
        if name=="T"; z = T.T.z end
        lines!(scene, f[name](T)[:,288].*scaling_factor[name], z, linewidth=3)
    end
    scene
end

function z(name, T)
    if name=="T"; return T.T.z; end
    return T.uw.z
end
lins = [lines!(f[name](T)[:,288].*scaling_factor[name], z(name, T), linewidth=3) for (file,T) in Ts]

# leg = legend(lins, files)
vbox(generate_scene("uw"), generate_scene("vw"), generate_scene("wT"), generate_scene("T"))
