using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using BenchmarkTools


train_files = ["-1e-3"]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

PATH = pwd()

OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(OUTPUT_PATH, "test_nonmutating.jld2")

FILE_PATH_NN = joinpath(PATH, "extracted_training_output", "NDE_training_convective_adjustment_1sim_-1e-3_2_extracted.jld2")

file = jldopen(FILE_PATH_NN, "r")
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

train_epochs = [10]
train_tranges = [1:40:1153]
train_optimizers = [[ADAM(0.01)] for i in 1:length(train_epochs)]
timestepper = ROCK4()

benchmark_mutating = []
benchmark_nonmutating = []

tranges = [1:10:100, 1:20:200, 1:10:200, 1:20:400, 1:40:400]


for trange in tranges
    @info "$(trange), mutating"
    t = @benchmark train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, $trange, ROCK4(), [ADAM(0.01)], 1, FILE_PATH, 1, 1, 10f0, 1)
    push!(benchmark_mutating, t)
    @info "$(trange), non-mutating"
    t = @benchmark train_NDE_convective_adjustment_nonmutating(uw_NN, vw_NN, wT_NN, 𝒟train, $trange, ROCK4(), [ADAM(0.01)], 1, FILE_PATH, 1, 1, 10f0, 1)
    push!(benchmark_nonmutating, t)
end

@info minimum.(benchmark_mutating)
@info minimum.(benchmark_nonmutating)
