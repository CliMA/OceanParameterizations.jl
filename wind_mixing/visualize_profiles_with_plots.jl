using Plots
using OceanParameterizations

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

Ts_reconstructed = Dict() # maps file name to 𝒟 struct
Ts = Dict()
for file in files
    Ts_reconstructed[file] = data(file, reconstruct_fluxes=true) # <: OceananigansData
    Ts[file] = data(file, reconstruct_fluxes=false) # <: OceananigansData
end

x_lims = Dict(
    "uw" => (-8,1),
    "vw" => (-1,4.5),
    "wT" => (-1.5,0.7),
    "T" => (19.6,20)
)

f = Dict(
    "uw" => 𝒟 -> 𝒟.uw.coarse,
    "vw" => 𝒟 -> 𝒟.vw.coarse,
    "wT" => 𝒟 -> 𝒟.wT.coarse,
    "T"  => 𝒟 -> 𝒟.T.coarse
)

zs = Dict(
    "uw" => 𝒟 -> 𝒟.uw.z,
    "vw" => 𝒟 -> 𝒟.vw.z,
    "wT" => 𝒟 -> 𝒟.wT.z,
    "T"  => 𝒟 -> 𝒟.T.z
)

# legend_placement = Dict(
#     "uw" => :bottomleft,
#     "vw" => :bottomright,
#     "wT" => :right
# )

legend_placement = Dict(
    "uw" => false,
    "vw" => false,
    "wT" => false,
    "T"  => false,
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

function plot_frame_i(name, Ts, i)
    p = plot(xlabel=x_labels[name], xlims = x_lims[name], ylabel="Depth (m)", palette=:Paired_6, legend=legend_placement[name], foreground_color_grid=:white, plot_titlefontsize=20)
    for (file, T) in Ts
        plot!(f[name](T)[:,i].*scaling_factor[name], zs[name](T), title = titles[name], label="$(file)", linewidth=3)
    end
    plot!(size=(400,500))
    p
end

p1 = plot_frame_i("uw", Ts, 288)
png(p1, pwd()*"/uw_last_frame.png")

p2 = plot_frame_i("vw", Ts, 288)
png(p2, pwd()*"/vw_last_frame.png")

p3 = plot_frame_i("wT", Ts, 288)
png(p3, pwd()*"/wT_last_frame.png")

pT = plot_frame_i("T", Ts, 288)
png(pT, pwd()*"/T_last_frame.png")

p4 = plot(grid=false, showaxis=false, palette=:Paired_6, ticks=nothing)
for file in files
    plot!(1, label=file_labels[file], legend=:left, size=(200,600))
end
p4
png(p4, pwd()*"/legend_last_frame.png")

layout = @layout [a b c d]
p = plot(p1,p2,p3,p4,layout=layout, size=(1600,400), tickfontsize=12)
png(p, pwd()*"/all_last_frame_new_suite.png")

layout = @layout [a b c d e]
p = plot(p1,p2,p3,pT,p4,layout=layout, size=(1600,400), tickfontsize=12)
png(p, pwd()*"/all_last_frame_new_suite_with_T.png")

## ANIMATION

function animate_all(name, Ts)
    anim = @animate for i in 1:288
        plot_frame_i(name, Ts, i)
    end
    return anim
end

save_animation(anim, filename) = gif(anim, pwd()*filename, fps=20)
# save_animation(animate_all("uw", Ts_reconstructed), "/uw_unscaled_reconstructed.gif")
# save_animation(animate_all("vw", Ts_reconstructed), "/vw_unscaled_reconstructed.gif")
# save_animation(animate_all("wT", Ts_reconstructed), "/wT_unscaled_reconstructed.gif")
save_animation(animate_all("uw", Ts), "/uw_unscaled.gif")
save_animation(animate_all("vw", Ts), "/vw_unscaled.gif")
save_animation(animate_all("wT", Ts), "/wT_unscaled.gif")
save_animation(animate_all("T", Ts), "/T_unscaled.gif")
