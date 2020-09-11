using Printf
using LinearAlgebra
using DiffEqFlux: FastLayer

struct ConservativeDiffusionLayer{C, K} <: FastLayer
    C :: C
    κ :: K

    function ConservativeDiffusionLayer(N, κ)
        # Define conservation matrix
        C = Matrix(1.0I, N, N)
        C[end, 1:end-1] .= -1
        C[end, end] = 0
        return new{typeof(C), typeof(κ)}(C, κ)
    end
end

(L::ConservativeDiffusionLayer)(u, p) = L.κ * L.C * u

function generate_neural_pde_architecture(N, κ; type)
    if type == :feed_forward
        dudt_NN = FastChain(FastDense(N, N))
    elseif type == :conservative_feed_forward
        dudt_NN = FastChain(FastDense(N, N), ConservativeDiffusionLayer(N, κ))
    end
    return dudt_NN
end

function train_diffusion_neural_pde!(npde, solutions, optimizers, epochs=1)
    time_steps = length(npde.kwargs[:saveat])
    ics = [sol[1] for sol in values(solutions)]
    correct_sols = [Array(sol) for sol in values(solutions)]

    loss(θ) = sum(Flux.mse(Array(npde(u₀, θ)), u_correct) for (u₀, u_correct) in zip(ics, correct_sols)) + 1e-3 * sum(abs, θ)

    function cb(θ, args...)
        @info @sprintf("Training free convection neural PDE... loss = %e", loss(θ))
        return false
    end

    for opt in optimizers
        if opt isa Optim.AbstractOptimizer
            for e in 1:epochs
                @info "Training free convection neural PDE for $(time_steps-1) time steps with $(typeof(opt)) [epoch $e/$epochs]..."
                res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2))
                display(res)
                npde.p .= res.minimizer
            end
        else
            for e in 1:epochs
                @info "Training free convection neural PDE for $(time_steps-1) time steps with $(typeof(opt))(η=$(opt.eta)) [epoch $e/$epochs]..."
                res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2), maxiters=1000)
                display(res)
                npde.p .= res.minimizer
            end
        end
    end

    return npde
end

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
