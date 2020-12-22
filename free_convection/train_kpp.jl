using DataDeps
using GeoData
using Flux
using JLD2
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

eki_ensemble_size = 10
eki_iterations = 10
eki, eki_loss_history = optimize_kpp_parameters(coarse_training_datasets, true_solutions, T_scaling,
                                                ensemble_members=eki_ensemble_size, iterations=eki_iterations)

plot(eki_loss_history, linewidth=3, linealpha=0.8, yaxis=:log, ylims=(1e-3, 1e-1),
     label="", xlabel="EKI iteration", ylabel="mean squared error",
     title="Optimizing KPP: EKI particle loss", grid=false, framestyle=:box, dpi=200)

savefig("eki_kpp_loss_history.png")

kwargs = (label="", bins=0:0.1:1.1, xlims=(0, 1.1), grid=false, framestyle=:box)

anim = @animate for n in 1:eki_iterations
    h1 = histogram(clamp.(eki.u[n][:, 1], 0, 1), xlabel="CSL", title="Optimizing KPP parameters"; kwargs...)
    h2 = histogram(clamp.(eki.u[n][:, 2], 0, 1), xlabel="CNL", title="EKI iteration $n"; kwargs...)
    h3 = histogram(clamp.(eki.u[n][:, 3], 0, 1), xlabel="CbT"; kwargs...)
    h4 = histogram(clamp.(eki.u[n][:, 4], 0, 1), xlabel="CKE"; kwargs...)
    plot(h1, h2, h3, h4, layout=(2, 2), dpi=200)
end

gif(anim, "eki_kpp_parameters_histograms.gif", fps=1)
