using DataDeps
using OceanParameterizations
using FreeConvection
using FreeConvection: coarse_grain

## Neural differential equation parameters

Nz = 32

# NN = Chain(Dense( Nz, 4Nz, relu),
#            Dense(4Nz, 4Nz, relu),
#            Dense(4Nz, Nz-1))

## Register data dependencies

for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

## Load training data

training_datasets = tds = Dict(
    1 => NCDstack(datadep"lesbrary_free_convection_1/statistics.nc"),
    2 => NCDstack(datadep"lesbrary_free_convection_2/statistics.nc")
)

## Add surface fluxes to data

training_datasets = tds = Dict(id => add_surface_fluxes(ds) for (id, ds) in tds)

## Coarse grain training data

coarse_training_datasets = ctds =
    Dict(id => coarse_grain(ds, Nz) for (id, ds) in tds)

## Create animations for T(z,t) and wT(z,t)

for id in keys(tds)
    T_filepath = "free_convection_T_$id"
    animate_variable(tds[id][:T], ctds[id][:T], xlabel="Temperature T (°C)", filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$id"
    animate_variable(tds[id][:wT], ctds[id][:wT], xlabel="Heat flux wT (m/s °C)", filepath=wT_filepath, frameskip=5)
end

## Pull out input (T) and output (wT) training data

itd = input_training_data(coarse_training_datasets)
otd = output_training_data(coarse_training_datasets)

## Feature scaling

T_training_data = cat([input.temperature for input in itd]..., dims=2)
wT_training_data = otd

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

itd = [rescale(i, T_scaling, wT_scaling) for i in itd]
otd = wT_scaling.(otd)

