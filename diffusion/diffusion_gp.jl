@gen function generate_gp_kernel()
    l ~ gamma(1, 2)
    σ² ~ gamma(1, 2)
    kernel = SquaredExponential(l, σ²)
    return kernel
end

@gen function train_diffusion_gp(x_train, y_train)
    kernel ~ generate_gp_kernel()
    return GaussianProcess(x_train, y_train, kernel)
end

@gen function predict_diffusion_gp(x_train, y_train, solutions)
    gp ~ train_diffusion_gp(x_train, y_train)

    for (name, sol) in solutions
        N, Nt = size(sol)
        u = sol.u[1]

        for n in 2:Nt
            u = predict(gp, [u])
            for i in 1:N
                {(:u, name, n, i)} ~ normal(u[i], 0.01)
            end
        end
    end

    return nothing
end

function infer_gp_hyperparameters(x_train, y_train, solutions; iters)
    observations = Gen.choicemap()

    for (name, sol) in solutions
        N, Nt = size(sol)
        u = sol.u[1]
        for n in 2:Nt, i in 1:N
            observations[(:u, name, n, i)] = sol.u[n][i]
        end
    end

    trace, _ = Gen.generate(predict_diffusion_gp, (x_train, y_train, solutions), observations)

    gp_hyperparameters = select(:gp => :kernel => :l, :gp => :kernel => :σ²)

    traces = []
    for _ in 1:iters
        trace, _ = metropolis_hastings(trace, gp_hyperparameters)
        push!(traces, trace)
    end

    return traces
end

function test_diffusion_gp(gp, solutions)
    for (name, sol) in solutions
        u_GP = animate_gp_test(sol, gp, filename="GP_test_$name.mp4")
        plot_conservation(u_GP, sol.t, filename="GP_conservation_$name.png")
    end
    return nothing
end

function animate_gp_test(sol, gp; filename, fps=15)
    N, Nt = size(sol)
    x = sol.prob.p.x

    u_GP = zeros(N, Nt)
    u_GP[:, 1] .= sol.u[1]

    for n in 2:Nt
        u_GP[:, n] .= predict(gp, [u_GP[:, n-1]])
    end

    anim = @animate for n=1:Nt
        plot(x, sol.u[n],    linewidth=2, ylim=(0, 2), label="Data", show=false)
        plot!(x, u_GP[:, n], linewidth=2, ylim=(0, 2), label="GP", show=false)
    end

    mp4(anim, filename, fps=fps)

    return u_GP
end
