using OrdinaryDiffEq
using DiffEqSensitivity
using Flux
using GalacticOptim
using OceanParameterizations

function DE_diffusion(u, p, t)
    N, κ = p

    D_face = Dᶠ(N, 1)
    D_cell = Dᶜ(N, 1)
    @inline ∂x²(x) = D_cell * D_face * x

    return κ * ∂x²(u)
end

N = 10
κ = 0.1f0

u₀ = rand(Float32, N)
t = 1:1:100f0
tspan = (t[1], t[end])

p = (N, κ)

prob = ODEProblem(DE_diffusion, u₀, tspan, p, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

sol = solve(prob, Rosenbrock23(), saveat=t)

function NDE(u, weights, re, t)
    NN = re(weights)
    return NN(u)
end

hidden_units = 50
weights, re = Flux.destructure(Chain(Dense(10, hidden_units, leakyrelu), Dense(hidden_units, 10)))

prob_NDE = ODEProblem((u, p, t) -> NDE(u, p, re, t), u₀, tspan, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

timestepper = Rosenbrock23()

sol_NDE = solve(prob_NDE, timestepper, saveat=t, p=weights)

function loss(weights, params)
    sol_NDE = Array(solve(prob_NDE, timestepper, saveat=t, p=weights))
    return Flux.mse(sol, sol_NDE)
end

loss(weights, nothing)

f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())

prob_loss = OptimizationProblem(f_loss, @view(weights[:]))

function cb(args...)
    @info args[2]
    false
end

opt = ADAM()
solve(prob_loss, opt, cb=cb, maxiters=3)

