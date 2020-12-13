file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_coriolis" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

zs = Dict(
    "u" => 𝒟 -> 𝒟.u.z, "v" => 𝒟 -> 𝒟.v.z, "T"  => 𝒟 -> 𝒟.T.z,
    "uw" => 𝒟 -> 𝒟.uw.z, "vw" => 𝒟 -> 𝒟.vw.z, "wT" => 𝒟 -> 𝒟.wT.z
)

scaling_factor = Dict(
     "u" => 1,
     "v" => 1,
     "T" => 1,
    "uw" => 1e4,
    "vw" => 1e4,
    "wT" => 1e4,
    "T" => 1
)

x_labels = Dict(
     "u" => "U (m/s)",
     "v" => "V (m/s)",
     "T" => "T (m/s)",
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

function animate_prediction(xs, name, 𝒟, test_file; filename=name, legend_labels=["" for i in 1:length(xs)], directory="Output")
    filepath = pwd() * "/" * directory * "/"
    isdir(dirname(filepath)) || mkpath(filepath)

    anim = @animate for n in 1:size(xs[1],2)
        x_max = maximum([maximum(x) for x in xs]).*scaling_factor[name]
        x_min = minimum([minimum(x) for x in xs]).*scaling_factor[name]

        fig = plot(xlim=(x_min, x_max), legend=:bottom, size=(400,400), xlabel=x_labels[name], ylabel="Depth (m)")
        for i in reverse(1:length(xs))
            plot!(fig, xs[i][:,n].*scaling_factor[name], zs[name](𝒟), label=legend_labels[i], title=file_labels[test_file]*", $(round(𝒟.t[n]/86400, digits=1)) days", linewidth=4, la=0.5, palette=:Set1_3)
        end

    end

    gif(anim, pwd() * "/$(directory)/$(filename).gif", fps=20)
end
