using Plots
using ClimateParameterizations

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis", "weak_wind_strong_cooling",
          "strong_wind_weak_cooling", "strong_wind_weak_heating"]




## T profiles

Ts = Dict() # maps file name to T array
Ts2 = Dict() # maps file name to T array
for file in files
    les = ClimateParameterizations.Data.read_les_output(file) # <: OceananigansData
    Ts[file] = ZeroMeanUnitVarianceScaling(les.T).(les.T)
    Ts2[file] = MinMaxScaling(les.T).(les.T)
end

Ts
Ts2

anim = @animate for i in 1:289
    plot(legend=false, xlabel="T", ylabel="z", xlims=(-1.7,2.3))
    for (file, T) in Ts
        plot!(T[:,i], collect(-255:2:0), title = "ZeroMeanUnitVarianceScaling", label="$(file)")
    end
end
gif(anim, pwd()*"/ZeroMeanUnitVarianceScaling.gif", fps=20)

anim = @animate for i in 1:289
    plot(legend=false, xlabel="T", ylabel="z", xlims=(0.0,1.0))
    for (file, T) in Ts
        plot!(T[:,i], collect(-255:2:0), title = "MinMaxScaling", label="$(file)")
    end
end
gif(anim, pwd()*"/MinMaxScaling.gif", fps=20)




## uw profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    les = ClimateParameterizations.Data.read_les_output(file) # <: OceananigansData
    Ts3[file] = les.wu
end

Ts3

anim = @animate for i in 1:289
    plot(legend=false, xlabel="uw", ylabel="z", xlims=(-0.0007,0.0001))
    for (file, T) in Ts3
        plot!(T[:,i], collect(-255:2:1), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/uw_unscaled.gif", fps=20)



## uw profiles

Ts = Dict() # maps file name to T array
Ts2 = Dict() # maps file name to T array
for file in files
    les = ClimateParameterizations.Data.read_les_output(file) # <: OceananigansData
    Ts[file] = ZeroMeanUnitVarianceScaling(les.wu).(les.wu)
    Ts2[file] = MinMaxScaling(les.wu).(les.wu)
end

Ts
Ts2

anim = @animate for i in 1:289
    plot(legend=false, xlabel="uw", ylabel="z", xlims=(-3,3))
    for (file, T) in Ts
        plot!(T[:,i], collect(-255:2:1), title = "ZeroMeanUnitVarianceScaling", label="$(file)")
    end
end
gif(anim, pwd()*"/ZeroMeanUnitVarianceScaling_uw.gif", fps=20)

anim = @animate for i in 1:289
    plot(legend=false, xlabel="uw", ylabel="z", xlims=(0,1))
    for (file, T) in Ts2
        plot!(T[:,i], collect(-255:2:1), title = "MinMaxScaling", label="$(file)")
    end
end
gif(anim, pwd()*"/MinMaxScaling_uw.gif", fps=20)

## vw profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    les = ClimateParameterizations.Data.read_les_output(file) # <: OceananigansData
    Ts3[file] = les.wv
end

Ts3

anim = @animate for i in 1:289
    plot(legend=false, xlabel="vw", ylabel="z", xlims=(-0.0001,0.0005))
    for (file, T) in Ts3
        plot!(T[:,i], collect(-255:2:1), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/vw_unscaled.gif", fps=20)


## wT profiles unscaled

Ts3 = Dict() # maps file name to T array
for file in files
    les = ClimateParameterizations.Data.read_les_output(file) # <: OceananigansData
    Ts3[file] = les.wT
end

Ts3

anim = @animate for i in 1:289
    plot(legend=false, xlabel="wT", ylabel="z", xlims=(-0.00009,0.00006))
    for (file, T) in Ts3
        plot!(T[:,i], collect(-255:2:1), title = "Unscaled", label="$(file)")
    end
end
gif(anim, pwd()*"/wT_unscaled.gif", fps=20)
