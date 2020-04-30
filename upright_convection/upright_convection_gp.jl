using Dates
using Printf
using Gen
using OceanTurb
using JLD2
using Plots
using ClimateSurrogates

import PyPlot
const plt = PyPlot

# For quick headless plotting without warnings.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Load Oceananigans free convection data from JLD2 file
#####

file = jldopen("free_convection_profiles.jld2")

Is = keys(file["timeseries/t"])

Nz = file["grid/Nz"]
Lz = file["grid/Lz"]
Nt = length(Is)

t = zeros(Nt)
T = T_data = zeros(Nt, Nz)
wT = zeros(Nt, Nz)

for (i, I) in enumerate(Is)
    t[i] = file["timeseries/t/$I"]
    T[i, :] = file["timeseries/T/$I"][1, 1, 2:Nz+1]
end

# Code credit: https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/utils.jl

"""
avg(Φ, n)
# Description
- Average a field down by n.
- Requires field to have evenly spaced points. Size of N leq length(Φ).
- Furthermore requires
# Arguments
- `Φ` :(vector) The field, an array
- `n` :(Int) number of grid points to average down to.
# Return
- `Φ2` :(vector) The field with values averaged, an array
"""
function avg(Φ, n)
    m = length(Φ)
    scale = Int(floor(m/n))
    if ( abs(Int(floor(m/n)) - m/n) > eps(1.0))
        return error
    end
    Φ2 = zeros(n)
    for i in 1:n
        Φ2[i] = 0
            for j in 1:scale
                Φ2[i] += Φ[scale*(i-1) + j] / scale
            end
    end
    return Φ2
end

@gen function generate_gp_kernel()
    l ~ gamma(1, 2)
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

function infer_gp_hyperparameters(x_train, y_train, x_test, y_test; iters)
    observations = Gen.choicemap()

    for (x, y) in zip(x_test, y_test)
        N, Nt = size(sol)
        for n in 2:Nt, i in 1:N
            observations[(:u, n, i)] = sol.u[n][i]
        end
    end

    trace, _ = Gen.generate(predict_convection_gp, (x_train, y_train, solutions), observations)

    gp_hyperparameters = select(:gp => :kernel => :l, :gp => :kernel => :σ²)

    traces = []
    for _ in 1:iters
        trace, _ = metropolis_hastings(trace, gp_hyperparameters)
        push!(traces, trace)
    end

    return traces
end

Nt, N = size(T)
coarse_resolution = cr = 16
Tₙ    = zeros(cr, Nt-1)
Tₙ₊₁  = zeros(cr, Nt-1)

for i in 1:Nt-1
      Tₙ[:, i] .=  avg(T[i, :], cr)
    Tₙ₊₁[:, i] .=  avg(T[i+1, :], cr)
end

n_train = round(Int, (Nt-1)/2)
training_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:n_train]
testing_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in n_train:Nt-1]

x_train = [data[1] for data in training_data]
y_train = [data[2] for data in training_data]

x_test = [data[1] for data in testing_data]
y_test = [data[2] for data in testing_data]
