include("upright_convection.jl")

@gen function generate_gp_kernel()
    l ~ gamma(1, 5000)
    σ² ~ gamma(1, 2)
    kernel = SquaredExponential(l, σ²)
    return kernel
end

@gen function train_convection_gp(x_train, y_train)
    kernel ~ generate_gp_kernel()
    return GaussianProcess(x_train, y_train, kernel)
end

@gen function predict_convection_gp(x_train, y_train, x_test, y_test)
    gp ~ train_convection_gp(x_train, y_train)

    Nt, N = length(x_test), length(x_test[1])

    u = x_test[1]
    for n in 2:Nt
        u = predict(gp, [u])
        for i in 1:N
            {(:u, n, i)} ~ normal(u[i], 0.01)
        end
    end

    return nothing
end

function infer_gp_hyperparameters(x_train, y_train, x_test, y_test; iters, verbose=true)
    observations = Gen.choicemap()

    Nt, N = length(x_test), length(x_test[1])
    for n in 2:Nt, i in 1:N
        observations[(:u, n, i)] = x_test[n][i]
    end

    gp_hyperparameters = select(:gp => :kernel => :l, :gp => :kernel => :σ²)
    trace, _ = Gen.generate(predict_convection_gp, (x_train, y_train, x_test, y_test), observations)
    accepts = 0

    for i in 1:iters
        trace, accepted = metropolis_hastings(trace, gp_hyperparameters, observations=observations)
        accepts += accepted
        if verbose
            @info "Iteration $i, acceptance ratio: " * @sprintf("%.4f", accepts/i)
        end
    end

    return trace
end

function test_convection_gp(gp, x_train, y_train, x_test, y_test)
    u = x_train[1]
    Nt, N = length(x_train) + length(x_test), length(u)

    us = []
    push!(us, u)

    for n in 2:Nt
        u = predict(gp, [u])
        push!(us, u)
    end

    return us
end

function animate_convection_gp(T, us)
    Nt, N = size(T)

    anim = @animate for n=1:5:Nt
        p = plot(T[n, :], zC, linewidth=2,
             xlim=(19, 20), ylim=(-100, 0), label="LES",
             xlabel="Temperature (C)", ylabel="Depth (z)",
             dpi=200, show=false)

        plot!(p, us[n], zC_cs, linewidth=2, label="GP", legend=:bottomright)
    end

    mp4(anim, "deepening_mixed_layer_GP.mp4", fps=15)
end

T, zC, t, Nz, Lz, constants, Q, FT, ∂T∂z = load_data("free_convection_profiles.jld2")

Nt, N = size(T)
coarse_resolution = cr = 16
Tₙ    = zeros(cr, Nt-1)
Tₙ₊₁  = zeros(cr, Nt-1)

zC_cs = coarse_grain(zC, cr)

for i in 1:Nt-1
      Tₙ[:, i] .=  coarse_grain(T[i, :], cr)
    Tₙ₊₁[:, i] .=  coarse_grain(T[i+1, :], cr)
end

# n_train = round(Int, (Nt-1)/2)
# training_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:n_train]
# testing_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in n_train:Nt-1]

n_train = 1:5:Nt-1
n_test = filter(n -> n ∉ n_train, 1:Nt-1)

training_data = [(Tₙ[:, n], Tₙ₊₁[:, n]) for n in n_train]
testing_data = [(Tₙ[:, n], Tₙ₊₁[:, n]) for n in n_test]

x_train = [data[1] for data in training_data]
y_train = [data[2] for data in training_data]

x_test = [data[1] for data in testing_data]
y_test = [data[2] for data in testing_data]

ls, σ²s = [], []

samples = 100
for n in 1:samples
    @info "Sample $n/$samples"
    trace = infer_gp_hyperparameters(x_train, y_train, x_test, y_test, iters=100)
    push!(ls, trace[(:gp => :kernel => :l)])
    push!(σ²s, trace[(:gp => :kernel => :σ²)])
end

bson_filename = "inferred_GP_hyperparameters.bson"
@info "Saving $bson_filename..."
bson(bson_filename, Dict(:l => ls, :σ² => σ²s))

# l = trace[(:gp => :kernel => :l)]
# σ² = trace[(:gp => :kernel => :σ²)]
# gp = GaussianProcess(x_train, y_train, SquaredExponential(l, σ²))
# us = test_convection_gp(gp, x_train, y_train, x_test, y_test)
# animate_convection_gp(T, us)
