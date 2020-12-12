using DataDeps
using GeoData
using NCDatasets
using FreeConvection
using FreeConvection: coarse_grain

## Neural differential equation parameters

Nz = 32

NN = Chain(Dense( Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

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
