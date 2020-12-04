using Plots
using ClimateParameterizations

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis", "weak_wind_strong_cooling",
          "strong_wind_weak_cooling", "strong_wind_weak_heating"]

## uw profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    𝒟 = ClimateParameterizations.Data.data(file, reconstruct_fluxes=false) # <: OceananigansData
    Ts3[file] = 𝒟.uw.coarse
end

Ts3

anim = @animate for i in 1:288
    # plot(legend=false, xlabel="uw", ylabel="z", xlims=(-0.0007,0.0001))
    plot(legend=false, xlabel="uw", ylabel="z", xlims=(-2e-4,1e-4))
    for (file, T) in Ts3
        # println(T[:,i])
        plot!(T[:,i], collect(1:33), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/uw_unscaled_les.gif", fps=20)

## vw profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    𝒟 = ClimateParameterizations.Data.data(file, reconstruct_fluxes=true) # <: OceananigansData
    Ts3[file] = 𝒟.vw.coarse
end

Ts3

anim = @animate for i in 1:289
    plot(legend=false, xlabel="vw", ylabel="z", xlims=(-0.0001,0.0005))
    for (file, T) in Ts3
        plot!(T[:,i], collect(-33:1), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/vw_unscaled.gif", fps=20)


## wT profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    𝒟 = ClimateParameterizations.Data.data(file, reconstruct_fluxes=true) # <: OceananigansData
    Ts3[file] = les.wT
end

Ts3

anim = @animate for i in 1:289
    plot(legend=false, xlabel="wT", ylabel="z", xlims=(-0.00009,0.00006))
    for (file, T) in Ts3
        plot!(T[:,i], collect(-33:1), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/wT_unscaled.gif", fps=20)
