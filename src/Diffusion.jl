module Diffusion

export generate_solutions,
       generate_neural_pde_architecture,
       train_diffusion_neural_pde!,
       animate_neural_pde_test

using Printf
using DifferentialEquations
using DiffEqFlux
# using Optim
using Plots

function diffusion!(∂u∂t, u, p, t)
    N, Δx, κ = p.N, p.Δx, p.κ
    @inbounds begin
        ∂u∂t[1] = κ * (u[N] -2u[1] + u[2]) / Δx^2
        for i in 2:N-1
            ∂u∂t[i] = κ * (u[i-1] -2u[i] + u[i+1]) / Δx^2
        end
        ∂u∂t[N] = κ * (u[N-1] -2u[N] + u[1]) / Δx^2
    end
    return nothing
end

"""
    solve_diffusion_equation(; u₀, N, L, κ, T, Nt)

Solve a 1D diffusion equation with initial condition given by `u₀(x)` on a domain -L/2 <= x <= L/2 with diffusivity `κ` using `N` grid points for time 0 <= t <= `T`. A solution with `Nt` outputs will be returned.
"""
function solve_diffusion_equation(; u₀, N, L, κ, T, Nt)
    Δx = L / N
    x = range(-L/2, L/2, length=N)
    ic = u₀.(x)

    tspan = (0.0, T)
    Δt = (tspan[2] - tspan[1]) / Nt
    t = range(tspan[1], tspan[2], length=Nt+1)

    params = (N=N, Δx=Δx, κ=κ, x=x)
    prob = ODEProblem(diffusion!, ic, tspan, params)
    solution = solve(prob, Tsit5(), saveat=t)

    return solution
end

function generate_solutions(training_functions, testing_functions; N, L, κ, T, Nt)
    solutions = []
    training_solutions = []
    testing_solutions = []

    for u₀ in (training_functions..., testing_functions...)
        sol = solve_diffusion_equation(u₀=u₀, N=N, L=L, κ=κ, T=T, Nt=Nt)
        push!(solutions, sol)

        if u₀ in training_functions
            push!(training_solutions, sol)
        elseif u₀ in testing_functions
            push!(testing_solutions, sol)
        end
    end

    return solutions, training_solutions, testing_solutions
end

function animate_solution(sol; filename, fps=15)
    Nt = length(sol)
    x = sol.prob.p.x

    anim = @animate for n=1:Nt
        plot(x, sol.u[n], linewidth=2, ylim=(0, 2), label="", show=false)
    end

    mp4(anim, filename, fps=fps)
end

function plot_conservation(u, t; filename)
    N, Nt = size(u)
    Σu₀ = sum(u[:, 1])
    Σu = [sum(u[:, n]) for n in 1:Nt]

    p = plot(t, Σu .- Σu₀, linewidth=2, title="Conservation", label="")

    @info "Saving $filename..."
    savefig(p, filename)
end

# function train_diffusion_neural_pde!(npde, solutions, optimizers, epochs=1)
#     time_steps = length(npde.kwargs[:saveat])
#     ics = [sol[1] for sol in values(solutions)]
#     correct_sols = [Array(sol) for sol in values(solutions)]

#     loss(θ) = sum(Flux.mse(Array(npde(u₀, θ)), u_correct) for (u₀, u_correct) in zip(ics, correct_sols)) + 1e-3 * sum(abs, θ)

#     function cb(θ, args...)
#         @info @sprintf("Training diffusion neural PDE... loss = %e", loss(θ))
#         return false
#     end

#     for opt in optimizers
#         if opt isa Optim.AbstractOptimizer
#             for e in 1:epochs
#                 @info "Training diffusion neural PDE for $(time_steps-1) time steps with $(typeof(opt)) [epoch $e/$epochs]..."
#                 res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2))
#                 display(res)
#                 npde.p .= res.minimizer
#             end
#         else
#             for e in 1:epochs
#                 @info "Training diffusion neural PDE for $(time_steps-1) time steps with $(typeof(opt))(η=$(opt.eta)) [epoch $e/$epochs]..."
#                 res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2), maxiters=1000)
#                 display(res)
#                 npde.p .= res.minimizer
#             end
#         end
#     end

#     return npde
# end

function test_diffusion_neural_pde(npde, solutions)
    for (name, sol) in solutions
        u_NN = animate_neural_pde_test(sol, npde, filename="NPDE_test_$name.mp4")
        plot_conservation(u_NN, sol.t, filename="NPDE_conservation_$name.png")
    end
    return nothing
end

function animate_neural_pde_test(sol, npde; filename, fps=15)
    N, Nt = size(sol)
    x = sol.prob.p.x
    u₀ = sol.u[1]

    u_NN = npde(u₀)

    anim = @animate for n=1:Nt
        plot(x, sol.u[n],    linewidth=2, ylim=(0, 2), label="Data", show=false)
        plot!(x, u_NN[:, n], linewidth=2, ylim=(0, 2), label="Neural PDE", show=false)
    end

    mp4(anim, filename, fps=fps)

    return u_NN
end

end