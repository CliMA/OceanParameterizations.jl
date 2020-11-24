using Statistics
using NCDatasets
using Plots
using Flux, DiffEqFlux, Optim
using ClimateParameterizations
using Oceananigans.Grids
using BSON

##
PATH = joinpath(pwd(), "wind_mixing")
DATA_PATH = joinpath(PATH, "Data", "wind_mixing_horizontal_averages_0.02Nm2_8days.nc")

ds = NCDataset(DATA_PATH)

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

function animate_gif(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str)
    PATH = joinpath(pwd(), "wind_mixing")
    anim = @animate for n in 1:size(xs[1],2)
    x_max = maximum(maximum(x) for x in xs)
    x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(uw,2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n]/86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    gif(anim, joinpath(PATH, "Output", "$(filename).gif"), fps=30)
end

# animate_gif([uw], zC, t, "uw")
# animate_gif([vw], zC, t, "vw")
# animate_gif([wT], zF, t, "wT")
# animate_gif([u], zC, t, "u")
# animate_gif([v], zC, t, "v")
# animate_gif([T], zC, t, "T")

function feature_scaling(x, mean, std)
    (x .- mean) ./ std
end

function reverse_scaling(x, mean, std)
    x .* std .+ mean
end


##
u_coarse = cat((coarse_grain(u[:,i], 32, Cell) for i in 1:size(u,2))..., dims=2)
v_coarse = cat((coarse_grain(v[:,i], 32, Cell) for i in 1:size(v,2))..., dims=2)
T_coarse = cat((coarse_grain(T[:,i], 32, Cell) for i in 1:size(T,2))..., dims=2)
uw_coarse = cat((coarse_grain(uw[:,i], 32, Cell) for i in 1:size(uw,2))..., dims=2)
vw_coarse = cat((coarse_grain(vw[:,i], 32, Cell) for i in 1:size(vw,2))..., dims=2)
wT_coarse = cat((coarse_grain_linear_interpolation(wT[:,i], 33, Face) for i in 1:size(wT,2))..., dims=2)
zC_coarse = coarse_grain(zC, 32, Cell)
zF_coarse = coarse_grain_linear_interpolation(zF, 33, Face)

uw_mean = mean(uw_coarse)
uw_std = std(uw_coarse)
vw_mean = mean(vw_coarse)
vw_std = std(vw_coarse)
wT_mean = mean(wT_coarse)
wT_std = std(wT_coarse)
u_mean = mean(u_coarse)
u_std = std(u_coarse)
v_mean = mean(v_coarse)
v_std = std(v_coarse)
T_mean = mean(T_coarse)
T_std = std(T_coarse)

uw_scaled = feature_scaling.(uw_coarse, uw_mean, uw_std)
vw_scaled = feature_scaling.(vw_coarse, vw_mean, vw_std)
wT_scaled = feature_scaling.(wT_coarse, wT_mean, wT_std)
u_scaled = feature_scaling.(u_coarse, u_mean, u_std)
v_scaled = feature_scaling.(v_coarse, v_mean, v_std)
T_scaled = feature_scaling.(T_coarse, T_mean, T_std)

uvT_scaled = cat(u_scaled, v_scaled, T_scaled, dims=1)
##
uw_train = [(uvT_scaled[:,i], uw_scaled[:,i]) for i in 1:size(uvT_scaled,2)]
vw_train = [(uvT_scaled[:,i], vw_scaled[:,i]) for i in 1:size(uvT_scaled,2)]
wT_train = [(uvT_scaled[:,i], wT_scaled[:,i]) for i in 1:size(uvT_scaled,2)]

model_uw = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,32))
loss_uw(x, y) = Flux.Losses.mse(model_uw(x), y)

function cb_uw()
    @info mean([loss_uw(uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train)])
    false
end

optimizers_uw = [ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent()]

for opt in optimizers_uw
    @info opt
    Flux.train!(loss_uw, params(model_uw), uw_train, opt, cb = Flux.throttle(cb_uw, 2))
end


uw_NN = (cat((reverse_scaling(model_uw(uw_train[i][1]), uw_mean, uw_std) for i in 1:length(uw_train))...,dims=2), uw_coarse)
animate_gif(uw_NN, zC_coarse, t, "uw", ["NN(u,v,T)", "truth"], "uw_NN")


model_vw = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,32))
loss_vw(x, y) = Flux.Losses.mse(model_vw(x), y)

optimizers_vw = [ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent()]

function cb_vw()
    @info mean([loss_vw(vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train)])
    false
end

for opt in optimizers_vw
    @info opt
    Flux.train!(loss_vw, params(model_vw), vw_train, opt, cb = Flux.throttle(cb_vw, 2))
end

vw_NN = (cat((reverse_scaling(model_vw(vw_train[i][1]), vw_mean, vw_std) for i in 1:length(vw_train))...,dims=2), vw_coarse)
animate_gif(vw_NN, zC_coarse, t, "vw", ["NN(u,v,T)", "truth"], "vw_NN")

model_wT = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,33))
loss_wT(x, y) = Flux.Losses.mse(model_wT(x), y)

function cb_wT()
    @info mean([loss_wT(wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train)])
    false
end

optimizers_wT = [ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent()]

for opt in optimizers_wT
    @info opt
    Flux.train!(loss_wT, params(model_wT), wT_train, opt, cb = Flux.throttle(cb_wT, 2))
end

wT_NN = (cat((reverse_scaling(model_wT(wT_train[i][1]), wT_mean, wT_std) for i in 1:length(wT_train))...,dims=2), wT_coarse)
animate_gif(wT_NN, zC_coarse, t, "wT", ["NN(u,v,T)", "truth"], "wT_NN")


uw_NN_params = Dict(
       :grid_points => 32,
    :neural_network => model_uw,
                :zC => zC_coarse,
        :u_scaling => u_scaled,
        :v_scaling => v_scaled,
         :T_scaling => T_scaled,
         :uvT_scaling => uvT_scaled,
        :uw_scaling => uw_scaled)

bson(joinpath(PATH, "Output","uw_NN_params.bson"), uw_NN_params)

vw_NN_params = Dict(
       :grid_points => 32,
    :neural_network => model_vw,
                :zC => zC_coarse,
        :u_scaling => u_scaled,
        :v_scaling => v_scaled,
         :T_scaling => T_scaled,
         :uvT_scaling => uvT_scaled,
        :vw_scaling => vw_scaled)

bson(joinpath(PATH, "Output","vw_NN_params.bson"), vw_NN_params)

wT_NN_params = Dict(
       :grid_points => 32,
    :neural_network => model_wT,
                :zF => zF_coarse,
        :u_scaling => u_scaled,
        :v_scaling => v_scaled,
         :T_scaling => T_scaled,
         :uvT_scaling => uvT_scaled,
        :wT_scaling => wT_scaled)

bson(joinpath(PATH, "Output","wT_NN_params.bson"), wT_NN_params)