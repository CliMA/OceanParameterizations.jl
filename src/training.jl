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