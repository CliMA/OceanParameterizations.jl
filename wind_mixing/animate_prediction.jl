file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_coriolis" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

# x_lims = Dict(
#     "uw" => (-0.0007,0.0001),
#     "vw" => (-0.0001,0.0005),
#     "wT" => (-0.00009,0.00006)
# )

zs = Dict(
    "uw" => 𝒟 -> 𝒟.uw.z, "vw" => 𝒟 -> 𝒟.vw.z, "wT" => 𝒟 -> 𝒟.wT.z, "T"  => 𝒟 -> 𝒟.T.z
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

# titles = Dict(
#     "uw" => "Zonal momentum flux, U'W'",
#     "vw" => "Meridional momentum flux, V'W'",
#     "wT" => "Temperature flux, W'T'",
#     "T" => "Temperature, T",
# )

function animate_prediction(xs, name, 𝒟, file; legend_labels=["" for i in 1:length(xs)], filename=name, directory="Output")
    filepath = pwd() * "/" * directory * "/"
    isdir(dirname(filepath)) || mkpath(filepath)

    anim = @animate for n in 1:length(𝒟.t)
        x_max = maximum([maximum(x) for x in xs]).*scaling_factor[name]
        x_min = minimum([minimum(x) for x in xs]).*scaling_factor[name]

        fig = plot(xlim=(x_min, x_max), legend=:bottom, size=(400,400))
        for i in reverse(1:length(xs))
            plot!(fig, xs[i][:,n].*scaling_factor[name], zs[name](𝒟), label=legend_labels[i], title=file_labels[file]*", $(round(𝒟.t[n]/86400, digits=1)) days", linewidth=4, la=0.5, palette=:Set1_3)
        end

        xlabel!(fig, x_labels[name])
        ylabel!(fig, "Depth (m)")
    end

    gif(anim, pwd() * "/$(directory)/$(filename).gif", fps=20)
end
