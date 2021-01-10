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

filepath = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "uw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")


uw_file = jldopen(filepath, "r")
uw_NN = uw_file["training_data/neural_network"]["$(length(keys(uw_file["training_data/neural_network"])))"]