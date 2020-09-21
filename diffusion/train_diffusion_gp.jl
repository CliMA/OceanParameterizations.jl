using LinearAlgebra
using Gen
using ClimateSurrogates

#####
##### Diffusion GP helper functions
#####

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

#####
##### Training and testing initial condition functions
#####

N = 16     # Number of grid points
L = 2      # Domain size -L/2 <= x <= L/2
κ = 1.5    # Diffusivity
T = 0.1    # Time span 0 <= t <= T
Nt = 32    # Number of time snapshots to save
Δt = T/Nt  # Time between outputs/snapshots

u₀_Gaussian(x) = exp(-50x^2)
u₀_quadratic(x) = 1 - x^2
u₀_sin(x) = 1 + sin(2π * x)
u₀_cos(x) = 1 - cos(2π * x)
u₀_shifted_sin(x) = 1 + sin(π * x + π/3)
u₀_shifted_cos(x) = 1 - cos(π * x - π/6)
u₀_zero(x) = 0.0
u₀_one(x) = 1.0

function_name(::typeof(u₀_Gaussian)) = "Gaussian"
function_name(::typeof(u₀_quadratic)) = "quadratic"
function_name(::typeof(u₀_sin))  = "sin"
function_name(::typeof(u₀_cos))  = "cos"
function_name(::typeof(u₀_shifted_sin))  = "shifted_sin"
function_name(::typeof(u₀_shifted_cos))  = "shifted_cos"
function_name(::typeof(u₀_zero)) = "zero"
function_name(::typeof(u₀_one)) = "one"

training_functions = (u₀_Gaussian, u₀_cos, u₀_shifted_sin, u₀_zero)
testing_functions = (u₀_quadratic, u₀_shifted_cos, u₀_sin, u₀_one)

solutions, training_solutions, testing_solutions, training_data, testing_data =
    generate_solutions(training_functions, testing_functions; N=N, L=L, κ=κ, T=T, Nt=Nt, animate=false)

x_train = [data[1] for data in training_data]
y_train = [data[2] for data in training_data]

#####
##### Train and test a Gaussian process
#####

gp = train_diffusion_gp(x_train, y_train)
test_diffusion_gp(gp, solutions)
