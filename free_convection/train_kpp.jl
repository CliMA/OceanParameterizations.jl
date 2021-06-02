using DataDeps
using Plots
using OceanParameterizations
using FreeConvection
using FreeConvection: coarse_grain

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["GKSwstype"] = "100"

Nz = 32

## Register data dependencies

@info "Registering data dependencies..."
for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

## Load data

@info "Loading data..."
datasets = Dict{Int,Any}(
    1 => NCDstack(datadep"free_convection_8days_Qb1e-8/statistics.nc"),
    2 => NCDstack(datadep"free_convection_8days_Qb2e-8/statistics.nc"),
    3 => NCDstack(datadep"free_convection_8days_Qb3e-8/statistics.nc"),
    4 => NCDstack(datadep"free_convection_8days_Qb4e-8/statistics.nc"),
    5 => NCDstack(datadep"free_convection_8days_Qb5e-8/statistics.nc"),
    6 => NCDstack(datadep"free_convection_8days_Qb6e-8/statistics.nc")
)

## Add surface fluxes to data

@info "Inserting surface fluxes..."
datasets = Dict{Int,Any}(id => add_surface_fluxes(ds) for (id, ds) in datasets)

## Coarse grain training data

@info "Coarse graining data..."
coarse_datasets = Dict{Int,Any}(id => coarse_grain(ds, Nz) for (id, ds) in datasets)

## Split into training and testing data

@info "Partitioning data into training and testing datasets..."

ids_train = [1, 2, 4, 6]
ids_test = [3, 5]

training_datasets = Dict(id => datasets[id] for id in ids_train)
testing_datasets = Dict(id => datasets[id] for id in ids_test)

coarse_training_datasets = Dict(id => coarse_datasets[id] for id in ids_train)
coarse_testing_datasets = Dict(id => coarse_datasets[id] for id in ids_test)

## Pull out input (T) and output (wT) training data

@info "Wrangling training data..."
input_training_data = wrangle_input_training_data(coarse_training_datasets)
output_training_data = wrangle_output_training_data(coarse_training_datasets)

## Feature scaling

@info "Scaling features..."

T_training_data = reduce(hcat, input.temperature for input in input_training_data)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

## Optimizing KPP parameters

@info "Optimizing KPP parameters..."
true_solutions = Dict(id => T_scaling.(ds[:T].data) for (id, ds) in coarse_datasets)

# eki_ensemble_size = 10
# eki_iterations = 10
# eki, eki_loss_history = optimize_kpp_parameters(coarse_training_datasets, true_solutions, T_scaling,
#                                                 ensemble_members=eki_ensemble_size, iterations=eki_iterations)

# plot(eki_loss_history, linewidth=3, linealpha=0.8, yaxis=:log,
#      label="", xlabel="EKI iteration", ylabel="mean squared error",
#      title="Optimizing KPP: EKI particle loss", grid=false, framestyle=:box, dpi=200)

# savefig("eki_kpp_loss_history.png")

# kwargs = (label="", grid=false, framestyle=:box)

# anim = @animate for n in 1:eki_iterations
#     h1 = histogram(eki.u[n][:, 1], bins=0:0.1:1, xlims=(0, 1), xlabel="CSL", title="Optimizing KPP parameters"; kwargs...)
#     h2 = histogram(eki.u[n][:, 2], bins=0:0.8:8, xlims=(0, 8), xlabel="CNL", title="EKI iteration $n"; kwargs...)
#     h3 = histogram(eki.u[n][:, 3], bins=0:0.6:6, xlims=(0, 6), xlabel="CbT"; kwargs...)
#     h4 = histogram(eki.u[n][:, 4], bins=0:0.5:5, xlims=(0, 5), xlabel="CKE"; kwargs...)
#     plot(h1, h2, h3, h4, layout=(2, 2), dpi=200)
# end

# gif(anim, "eki_kpp_parameters_histograms.gif", fps=1)

using Random, Distributions, LinearAlgebra
# numerical gradient
function gradient(L, params; δ = 1e-4 .* ones(length(params)))
    ∇L = zeros(length(params))
    e = I + zeros(length(params), length(params))
    Lcurrent = L(params)
    for i in eachindex(params)
        up = L(params + δ[i] * e[i,:])
        ∇L[i] = (up - Lcurrent)/(δ[i])
    end
    return ∇L
end

# Define Method structs
struct RandomPlugin{S,T,V,U}
    priors::S
    fcalls::T
    seed::V
    progress::U
end

function RandomPlugin(priors, fcalls::Int; seed = 1234, progress = true)
    return RandomPlugin(priors, fcalls, 1234, true)
end

struct RandomLineSearch{I, T, B}
    linesearches::I
    linelength::I
    linebounds::T
    progress::B
    seed::I
end

function RandomLineSearch(linesearches, linelength)
    return RandomLineSearch(linesearches, linelength, (-0.1,1), true, 1234)
end
function RandomLineSearch(; linesearches = 10, linelength = 10, linebounds = (-0.1,1), progress = true, seed = 1234)
    return RandomLineSearch(linesearches, linelength, linebounds, progress, seed)
end

# Define Helper functions

# RandomPlugin
function priorloss(ℒ, fcalls, priors; seed = 1234, progress = true)
    Random.seed!(seed)
    losses = []
    vals = []
    for i in 1:fcalls
        guessparams = rand.(priors)
        push!(vals, guessparams)
        currentloss = ℒ(guessparams)
        push!(losses, currentloss)
        if progress
            println("iteration " * string(i))
        end
    end
    return losses, vals
end

# LineSearch
function randomlinesearch(ℒ, ∇ℒ, bparams; linesearches = 10, linelength = 10, linebounds = (-0.1, 1.0), progress = true, seed = 1234)
    params = copy(bparams)
    Random.seed!(seed)
    for i in 1:linesearches
        αs = [rand(Uniform(linebounds...), linelength-1)..., 0.0]
        ∇L = ∇ℒ(params)
        LS = [ℒ(params - α * ∇L) for α in αs]
        b = argmin(LS)
        if progress
            println("best after line search ", i)
        end
        params .= params - αs[b] * ∇L
        if αs[b] == 0
            αs .= 0.5 .* αs
            if progress
                println("staying the same at ")
            end
        end
        if progress
            println((params, LS[b]))
        end
    end
    return params
end

# Define optimize functions
function optimize(ℒ, method::RandomPlugin; history = false, printresult = true)
    losses, vals = priorloss(ℒ, method.fcalls,
                       method.priors, seed = method.seed,
                       progress = method.progress)
    indmin = argmin(losses)
    if printresult
        println("The minimum loss is ", losses[indmin])
        println("The minimum argument is ", vals[indmin])
        println("This occured at iteration ", indmin)
    end
    if history == false
        return vals[indmin]
    else
        return vals, losses
    end
end
optimize(ℒ, initialguess, method::RandomPlugin; history = false, printresult = true) = optimize(ℒ, method::RandomPlugin; history = false, printresult = true)

function optimize(ℒ, ∇ℒ, params, method::RandomLineSearch; history = false, printresult = true)
    bestparams = randomlinesearch(ℒ, ∇ℒ, params; linesearches = method.linesearches,
                    linelength = method.linelength,
                    linebounds = method.linebounds,
                    progress = method.progress,
                    seed = method.seed)
    return bestparams
end

using Flux: mse
import OceanTurb

## Example
# loss function

# loss(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4]-4)^2

# loss(p) = mean(
#     mse(T_scaling.(free_convection_kpp(ds, parameters=OceanTurb.KPP.Parameters(CSL=p[1], CNL=p[2], Cb_T=p[3], CKE=p[4])).T),
#         true_solutions[id])
#     for (id, ds) in datasets
# )

loss(p) = mse(T_scaling.(free_convection_kpp(coarse_datasets[6], parameters=OceanTurb.KPP.Parameters(CSL=p[1], CNL=p[2], Cb_T=p[3], CKE=p[4])).T),
                         true_solutions[6])

# First construct global search
# Create Prior
lower = [0.0, 0.0, 0.0,  0.0]
upper = [1.0, 10.0, 10.0, 10.0]
priors = Uniform.(lower, upper)
# Determine number of function calls
functioncalls = 1000
# Define Method
method = RandomPlugin(priors, functioncalls)
# Optimize
minparam = optimize(loss, method)

# Next do gradient descent
# construct numerical gradient
∇loss(params) = gradient(loss, params)
# optimize choosing minimum from the global search for refinement
best_params = minparam
method  = RandomLineSearch(linebounds = (0, 1e-0/norm(∇loss(best_params))), linesearches = 20)
bestparam = optimize(loss, ∇loss, best_params, method)
