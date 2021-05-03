using Flux, CUDA
using Plots
using OrdinaryDiffEq
using BenchmarkTools

function truth_ODE!(du, u, p, t)
    A = cu([-0.1 2.; -2. -0.1])
    du .= A * u
end

u₀ = [2., 0.] |> gpu
trange = 0:0.1:1
tspan = (trange[1], trange[end])

truth_prob = ODEProblem(truth_ODE!, u₀, tspan, saveat=trange)
truth_timeseries = solve(truth_prob, Tsit5())

@btime solve(truth_prob, Tsit5())

function NDE(u, p, t)
    A = 
end


m = Dense(10,5) |> gpu

x = rand(10) |> gpu

using CUDA
using BenchmarkTools

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
@btime loss(cu(rand(5)), cu(rand(2))) # ~ 3

using JLD2

file = jldopen("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output\\parameter_optimisation_modified_pacanowski_philander.jld2", "r")
