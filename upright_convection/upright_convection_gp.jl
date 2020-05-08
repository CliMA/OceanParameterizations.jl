include("upright_convection.jl")

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

ls, σ²s, solutions = [], [], []

samples = 200
for n in 1:samples
    @info "Sample $n/$samples"
    trace = infer_gp_hyperparameters(x_train, y_train, x_test, y_test, iters=100)

    l, σ² = trace[(:gp => :kernel => :l)], trace[(:gp => :kernel => :σ²)]
    push!(ls, l)
    push!(σ²s, σ²)

    # Generate solution
    gp = GaussianProcess(x_train, y_train, SquaredExponential(l, σ²))
    sol = test_convection_gp(gp, x_train, y_train, x_test, y_test)
    push!(solutions, sol)
end

bson_filename = "inferred_GP_hyperparameters.bson"
@info "Saving $bson_filename..."

data = Dict(:l => ls, :σ² => σ²s, :solutions => solutions,
            :T => T, :zC => zC, :zC_cs => zC_cs)
bson(bson_filename, data)
