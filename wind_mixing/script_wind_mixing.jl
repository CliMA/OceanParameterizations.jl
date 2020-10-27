using Statistics
using NCDatasets
using Plots
using Flux, DiffEqFlux
# using ClimateSurrogates
using Oceananigans.Grids

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


function animate_gif(x, y, t, x_str)
    PATH = joinpath(pwd(), "wind_mixing")
    anim = @animate for n in 1:size(x,2)
        @info "$x_str frame of $n/$(size(uw,2))"
        plot(x[:,n], y, label=nothing, title="t = $(round(t[n]/86400, digits=2)) days",
             xlim=(minimum(x), maximum(x)), ylim=(minimum(y), maximum(y)))
        xlabel!("$x_str")
        ylabel!("z")
    end
    gif(anim, joinpath(PATH, "Output", "$(x_str).gif"), fps=30)
end

animate_gif(uw, zC, t, "uw")
animate_gif(vw, zC, t, "vw")
animate_gif(wT, zF, t, "wT")
animate_gif(u, zC, t, "u")
animate_gif(v, zC, t, "v")
animate_gif(T, zC, t, "T")


function coarse_grain(Φ, n, ::Type{Cell})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

function coarse_grain(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = (N-2) / (n-2)
    if isinteger(Δ)
        Φ̅[1], Φ̅[n] = Φ[1], Φ[N]
        Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Cell)
    else
        Φ̅[1], Φ̅[n] = Φ[1], Φ[N]
        for i in 2:n-1
            i1 = round(Int, 2 + (i-2)*Δ)
            i2 = round(Int, 2 + (i-1)*Δ)
            Φ̅[i] = mean(Φ[i1:i2])
        end
    end
    return Φ̅
end

model_uw = Chain(Dense(32,30, tanh), Dense(30,32))


loss_uw(x, y) = Flux.Losses.mse(model_uw(x), y)
p_uw = params(model_uw)


data_train_u = cat((coarse_grain(u[:,i], 32, Cell) for i in 1:size(u,2))..., dims=2)
data_train_uw = cat((coarse_grain(uw[:,i], 32, Cell) for i in 1:size(uw,2))..., dims=2)

# uw_data_x = [[u[i], v[i], T[i]] for i=1:length(u)]
# uw_data_train = zip(data_train_u, data_train_uw)

uw_train = [(data_train_u[:,i], data_train_uw[:,i]) for i in 1:size(data_train_u,2)]

function cb()
    @info mean([loss(uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train)])
end

Flux.train!(loss_uw, p_uw, uw_train, Descent(), cb = cb)
params(model_uw)
