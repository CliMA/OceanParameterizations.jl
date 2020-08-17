using LinearAlgebra
using DiffEqFlux: FastLayer

struct ConservativeDiffusionLayer{C, K} <: FastLayer
    C :: C
    κ :: K

    function ConservativeDiffusionLayer(N, κ)
        # Define conservation matrix.
        @show N
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

function train_diffusion_neural_pde(training_data, NN, optimizers, Δt)
    # Set up neural differential equation
    tspan_npde = (0.0, Δt)
    diffusion_npde = NeuralODE(NN, tspan_npde, Tsit5(), reltol=1e-4, saveat=[Δt])

    loss_function(θ, uₙ, uₙ₊₁) = Flux.mse(uₙ₊₁, diffusion_npde(uₙ, θ)) ./ (sum(abs2, uₙ₊₁) + 1e-6)
    training_loss(θ, data) = sum([loss_function(θ, data[i]...) for i in 1:length(data)])

    function cb(θ, args...)
        println("train_loss = $(training_loss(θ, training_data))")
        return false
    end

    # Train!
    for opt in optimizers
        @info "Training with optimizer: $(typeof(opt))..."
        if opt isa Optim.AbstractOptimizer
            full_loss(θ) = training_loss(θ, training_data)
            res = DiffEqFlux.sciml_train(full_loss, diffusion_npde.p, opt, cb=cb, maxiters=100)
            display(res)
            diffusion_npde.p .= res.minimizer
        else
            epochs = 4
            for e in 1:epochs
                @info "Training with optimizer: $(typeof(opt)) epoch $e..."
                res = DiffEqFlux.sciml_train(loss_function, diffusion_npde.p, opt, training_data, cb=cb)
                diffusion_npde.p .= res.minimizer
            end
        end
    end

    return diffusion_npde
end

function test_diffusion_neural_pde(npde, solutions)
    for (name, sol) in solutions
        u_NN = animate_neural_pde_test(sol, npde, filename="NPDE_test_$name.mp4")
        plot_conservation(u_NN, sol.t, filename="NPDE_conservation_$name.png")
    end
    return nothing
end

function animate_neural_pde_test(sol, nde; filename, fps=15)
    N, Nt = size(sol)
    x = sol.prob.p.x

    u_NN = zeros(N, Nt)
    u_NN[:, 1] .= sol.u[1]

    for n in 2:Nt
        sol_NN = nde(u_NN[:, n-1])
        u_NN[:, n] .= sol_NN.u[1]
    end

    anim = @animate for n=1:Nt
        plot(x, sol.u[n],    linewidth=2, ylim=(0, 2), label="Data", show=false)
        plot!(x, u_NN[:, n], linewidth=2, ylim=(0, 2), label="Neural PDE", show=false)
    end

    mp4(anim, filename, fps=fps)

    return u_NN
end
