using NCDatasets
using Plots
using Flux, DiffEqFlux

##
PATH = joinpath(pwd(), "wind_mixing")
DATA_PATH = joinpath(PATH, "Data", "wind_mixing_horizontal_averages_0.02Nm2_8days.nc")

ds = NCDataset(DATA_PATH)
keys(ds)


xC = Array(ds["xC"])
xF = Array(ds["xF"])
yC = Array(ds["yC"])
yF = Array(ds["yF"])
zC = Array(ds["zC"])
zF = Array(ds["zF"])

uT = Array(ds["uT"])
vT = Array(ds["vT"])
wT = Array(ds["wT"])

uu = Array(ds["uu"])
vv = Array(ds["vv"])
ww = Array(ds["ww"])
uv = Array(ds["uv"])
uw = Array(ds["uw"])
vw = Array(ds["vw"])

u = Array(ds["u"])
v = Array(ds["v"])

T = Array(ds["T"])
t = Array(ds["time"])
##
plot(T[:,end], zC)

uw_max = maximum(uw)
uw_min = minimum(uw)

anim_uw = @animate for n in 1:size(uw,2)
    @info "uw frame of $n/$(size(uw,2))"
    plot(uw[:,n], zC, label=nothing, title="t = $(round(Int64, t[n]))s",
         xlim=(uw_min, uw_max), ylim=(minimum(zC), maximum(zC)))
    xlabel!("uw")
    ylabel!("z")
end
gif(anim_uw, joinpath(PATH, "Output", "uw.gif"), fps=30)

vw_max = maximum(vw)
vw_min = minimum(vw)

anim_vw = @animate for n in 1:size(vw,2)
    @info "vw frame of $n/$(size(vw,2))"
    plot(vw[:,n], zC, label=nothing, title="t = $(round(Int64, t[n]))s",
         xlim=(vw_min, vw_max), ylim=(minimum(zC), maximum(zC)))
    xlabel!("vw")
    ylabel!("z")
end
gif(anim_vw, joinpath(PATH, "Output", "vw.gif"), fps=30)

wT_max = maximum(wT)
wT_min = minimum(wT)

anim_wT = @animate for n in 1:size(wT,2)
    @info "wT frame of $n/$(size(wT,2))"
    plot(wT[:,n], zF, label=nothing, title="t = $(round(Int64, t[n]))s",
         xlim=(wT_min, wT_max), ylim=(minimum(zF), maximum(zF)))
    xlabel!("wT")
    ylabel!("z")
end
gif(anim_wT, joinpath(PATH, "Output", "wT.gif"), fps=30)


model_uw = Chain(Dense(3,30, tanh), Dense(30,1))
loss_uw(x, y) = Flux.Losses.mse(model_uw(x), y)
p_uw = params(model_uw)


uw_data_x = [[u[i], v[i], T[i]] for i=1:length(u)]
uw_data_train = zip(uw_data_x, uw[:])


Flux.train!(loss, p_uw, uw_data_train, Descent(), cb=throttle())
params(model_uw)
