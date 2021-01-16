using OrdinaryDiffEq, DiffEqSensitivity
using Flux
using GalacticOptim

t_train = 0:0.1:1
tspan = (t_train[1], t_train[end])
u0 = [1.0, 1.0]

function ODE!(du, u, p, t)
    A = [-0.1 2.0; -2.0 -0.1]
    du .= A * u
end
prob = ODEProblem(ODE!, u0, tspan)
sol = Array(solve(prob, Tsit5(), saveat = t_train))

NN = Chain(Dense(2, 50, tanh), Dense(50, 50, tanh), Dense(50, 2))
weights, re = Flux.destructure(NN)

function NDE!(du, u, p, t)
    du .= re(p)(u)
end

prob_NN = ODEProblem(NDE!, u0, tspan)
sol_NN = solve(prob_NN, Tsit5(), p = weights, saveat = t_train)


function loss(weights, p)
    return Flux.mse(Array(solve(prob_NN, Tsit5(), p = weights, saveat = t_train)), sol)
end

loss(weights, nothing)

f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

loss_prob = OptimizationProblem(f, weights)

function cb(args...)
    @info args[2]
    false
end


@info loss(weights, nothing)
res = solve(loss_prob, ADAM(), cb = cb, maxiters = 5)M(), cb=cb, maxiters=5)

using FileIO
using JLD2
using Flux
using OceanParameterizations
using OrdinaryDiffEq, DiffEqSensitivity

filepath_input = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "test_input.jld2")
filepath_output = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "test_output.jld2")
    
NN = Chain(Dense(2,10, relu), Dense(10, 2))


using WindMixing
write_metadata_NN_training(filepath_input, ["1e-3"], [1, 2], [ADAM(), Descent()], NN, "uw")
data(["-1e-3"], scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
write_data_NN(filepath_input, NN, NN, NN)
a = load(filepath_input, "training_info/optimizers")

a["train_epochs"]

@load filepath_input "training_data"

a = jldopen(filepath_input, "r") do file
    load(file, "training_info")
end

a["train_epochs"]
jldopen()
a["train_epochs"]

uw_file = jldopen(filepath, "r")
uw_NN = uw_file["training_data/neural_network"]["$(length(keys(uw_file["training_data/neural_network"])))"]