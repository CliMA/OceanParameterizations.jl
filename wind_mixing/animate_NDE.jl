using Statistics
using NCDatasets
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")

train_files = ["strong_wind"]
output_gif_directory = "Output"

PATH = pwd()

𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")

function predict_NDE(NN, x, top, bottom)
    interior = NN(x)
    return [top; interior; bottom]
end

f = 1f-4
H = Float32(abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1]))
τ = Float32(abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1]))
Nz = 32
u_scaling = 𝒟train.scalings["u"]
v_scaling = 𝒟train.scalings["v"]
T_scaling = 𝒟train.scalings["T"]
uw_scaling = 𝒟train.scalings["uw"]
vw_scaling = 𝒟train.scalings["vw"]
wT_scaling = 𝒟train.scalings["wT"]
μ_u = Float32(u_scaling.μ)
μ_v = Float32(v_scaling.μ)
σ_u = Float32(u_scaling.σ)
σ_v = Float32(v_scaling.σ)
σ_T = Float32(T_scaling.σ)
σ_uw = Float32(uw_scaling.σ)
σ_vw = Float32(vw_scaling.σ)
σ_wT = Float32(wT_scaling.σ)

uw_top = Float32(𝒟train.uw.scaled[1,1])
uw_bottom = Float32(uw_scaling(-1e-3))
vw_top = Float32(𝒟train.vw.scaled[1,1])
vw_bottom = Float32(𝒟train.vw.scaled[end,1])
wT_top = Float32(𝒟train.wT.scaled[1,1])
wT_bottom = Float32(𝒟train.wT.scaled[end,1])

# uw_NN_model = BSON.load(joinpath(PATH, "Output", "uw_NN_params_2DaySuite.bson"))[:neural_network]
# vw_NN_model = BSON.load(joinpath(PATH, "Output", "vw_NN_params_2DaySuite.bson"))[:neural_network]
# wT_NN_model = BSON.load(joinpath(PATH, "Output", "wT_NN_params_2DaySuite.bson"))[:neural_network]
##
uw_NDE = BSON.load(joinpath(PATH, "Output", "uw_NDE_1sim_200.bson"))[:neural_network]
vw_NDE = BSON.load(joinpath(PATH, "Output", "vw_NDE_1sim_200.bson"))[:neural_network]
wT_NDE = BSON.load(joinpath(PATH, "Output", "wT_NDE_1sim_200.bson"))[:neural_network]

uw_weights, re_uw = Flux.destructure(uw_NDE)
vw_weights, re_vw = Flux.destructure(vw_NDE)
wT_weights, re_wT = Flux.destructure(wT_NDE)

# uw_weights = BSON.load(joinpath(PATH, "Output", "uw_NDE_weights_2DaySuite.bson"))[:weights]
# vw_weights = BSON.load(joinpath(PATH, "Output", "vw_NDE_weights_2DaySuite.bson"))[:weights]
# wT_weights = BSON.load(joinpath(PATH, "Output", "wT_NDE_weights_2DaySuite.bson"))[:weights]

# uw_weights = BSON.load(joinpath(PATH, "Output", "uw_NDE_weights_2DaySuite.bson"))[:weights]
# vw_weights = BSON.load(joinpath(PATH, "Output", "vw_NDE_weights_2DaySuite.bson"))[:weights]
# wT_weights = BSON.load(joinpath(PATH, "Output", "wT_NDE_weights_2DaySuite.bson"))[:weights]

size_uw_NN = length(uw_weights)
size_vw_NN = length(vw_weights)
size_wT_NN = length(wT_weights)

p_nondimensional = [f; τ; H; μ_u; μ_v; σ_u; σ_v; σ_T; σ_uw; σ_vw; σ_wT; uw_top; uw_bottom; vw_top; vw_bottom; wT_top; wT_bottom; uw_weights; vw_weights; wT_weights]


D_cell = Float32.(Dᶜ(Nz, 1/Nz))

function NDE_nondimensional_flux(x, p, t)
    f, τ, H, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = p[1:17]
    Nz = 32
    uw_weights = p[18:18+size_uw_NN-1]
    vw_weights = p[18+size_uw_NN:18+size_uw_NN+size_vw_NN-1]
    wT_weights = p[18+size_uw_NN+size_vw_NN:18+size_uw_NN+size_vw_NN+size_wT_NN-1]
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    A = - τ / H
    B = f * τ
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:96]
    dx₁ = A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NN, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
    dx₂ = A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NN, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
    dx₃ = A .* σ_wT ./ σ_T .* D_cell * predict_NDE(wT_NN, x, wT_top, wT_bottom)
    return [dx₁; dx₂; dx₃]
end

function time_window(t, uvT, trange)
    return (Float32.(t[trange]), Float32.(uvT[:,trange]))
end

start_index = 1
end_index = 289

timesteps = start_index:1:end_index
uvT₀ = Float32.(𝒟train.uvT_scaled[:,start_index])

t_train, uvT_train = time_window(𝒟train.t, 𝒟train.uvT_scaled, timesteps)
t_train = Float32.(t_train ./ τ)
tspan_train = (t_train[1], t_train[end])


opt_NDE = ROCK4()
# prob = ODEProblem(NDE_nondimensional_flux!, uvT₀, tspan_train, p_nondimensional, saveat=t_train)
prob = ODEProblem(NDE_nondimensional_flux, uvT₀, tspan_train, p_nondimensional, saveat=t_train)
sol = solve(prob, opt_NDE)

t_plots = 𝒟train.t[timesteps]
u_plots = (Array(sol)[1:32, :], uvT_train[1:32, :])
v_plots = (Array(sol)[33:64, :], uvT_train[33:64, :])
T_plots = (Array(sol)[65:96, :], uvT_train[65:96, :])

##
function animate_NDE(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str)
    PATH = joinpath(pwd(), "Output")
    anim = @animate for n in 1:size(xs[1],2)
    x_max = maximum(maximum(x) for x in xs)
    x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1],2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n]/86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
end

animate_NDE(u_plots, 𝒟train.u.z, t_plots, "u", ["NDE", "truth"], "u_NDE_2DaySuite_large_long_ROCK4")
animate_NDE(v_plots, 𝒟train.v.z, t_plots, "v", ["NDE", "truth"], "v_NDE_2DaySuite_large_long_ROCK4")
animate_NDE(T_plots, 𝒟train.v.z, t_plots, "T", ["NDE", "truth"], "T_NDE_2DaySuite_large_long_ROCK4")

# animate_NDE(u_plots, 𝒟train.u.z, t_plots, "u", ["NDE", "truth"], "u_NDE_2DaySuite_long")
# animate_NDE(v_plots, 𝒟train.v.z, t_plots, "v", ["NDE", "truth"], "v_NDE_2DaySuite_long")
# animate_NDE(T_plots, 𝒟train.v.z, t_plots, "T", ["NDE", "truth"], "T_NDE_2DaySuite_long")