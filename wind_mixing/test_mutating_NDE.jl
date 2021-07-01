using Statistics
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing
using JLD2
using FileIO
using WindMixing: prepare_parameters_NDE_training
using WindMixing: prepare_BCs
using BenchmarkTools
using LinearAlgebra


PATH = joinpath(pwd(), "extracted_training_output")
DATA_NAME = "NDE_training_mpp_10sim_windcooling_windheating_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4"
DATA_PATH = joinpath(PATH, "$(DATA_NAME)_extracted.jld2")

# file = jldopen(DATA_PATH, "r")
file = jldopen(DATA_PATH, "r")

train_files = file["training_info/train_files"]
train_parameters = file["training_info/parameters"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

close(file)
[uw_NN(rand(96)) uw_NN(rand(96))]

constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(
    𝒟train, uw_NN, vw_NN, wT_NN, 1f-4, 32, 9.81, 1.67f-4, 1f-4, 1f-1, 0.25f0, 0.1f0, 1f0, 10, (modified_pacanowski_philander=true, something=false))

test_files = ["-1e-3"]
𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
BCs = prepare_BCs(𝒟test, scalings)
uvT₀ = [scalings.u.(𝒟test.u.coarse[:,1]); scalings.v.(𝒟test.v.coarse[:,1]); scalings.T.(𝒟test.T.coarse[:,1])]
trange = 1:1:1153
ts = 𝒟test.t[trange] ./ constants.τ
timestepper = ImplicitEuler(autodiff=false)

Threads.nthreads() = 6
BLAS.set_num_threads(4)

@info "CPU"
sol = solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper)
# @btime solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper)
# @btime solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper)


@info "CPU 40 threads"
BLAS.set_num_threads(40)
# sol = solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper=timestepper)
@btime solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper=timestepper)
@btime solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper=timestepper)

# plot(sol[65:end, end], -31:1:0)


using CUDA
using WindMixing: solve_NDE_mutating_GPU
CUDA.allowscalar(false)

test_files = ["-1e-3"]
𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
BCs = prepare_BCs(𝒟test, scalings)
uvT₀_gpu = [scalings.u.(𝒟test.u.coarse[:,1]); scalings.v.(𝒟test.v.coarse[:,1]); scalings.T.(𝒟test.T.coarse[:,1])] |> gpu
trange = 1:1:1153
ts = 𝒟test.t[trange] ./ constants.τ
tspan_gpu = (ts[1], ts[end])
ts_gpu = ts
timestepper=ROCK4()

file = jldopen(DATA_PATH, "r")
uw_NN = file["neural_network/uw"] |> gpu
vw_NN = file["neural_network/vw"] |> gpu
wT_NN = file["neural_network/wT"] |> gpu
close(file)

uw_NN_gpu = uw_NN |> gpu
vw_NN_gpu = vw_NN |> gpu
wT_NN_gpu = wT_NN |> gpu

derivatives_gpu = (face=derivatives.face |> gpu, cell=derivatives.cell |> gpu)

@info "GPU"
# sol_gpu = solve_NDE_mutating_GPU(uw_NN_gpu, vw_NN_gpu, wT_NN_gpu, scalings, constants, BCs, derivatives_gpu, uvT₀_gpu, ts_gpu, tspan_gpu, timestepper=timestepper)
@btime solve_NDE_mutating_GPU(uw_NN_gpu, vw_NN_gpu, wT_NN_gpu, scalings, constants, BCs, derivatives_gpu, uvT₀_gpu, ts_gpu, tspan_gpu, timestepper=timestepper)
@btime solve_NDE_mutating_GPU(uw_NN_gpu, vw_NN_gpu, wT_NN_gpu, scalings, constants, BCs, derivatives_gpu, uvT₀_gpu, ts_gpu, tspan_gpu, timestepper=timestepper)
