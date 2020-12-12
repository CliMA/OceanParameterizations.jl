using DataDeps
using GeoData
using NCDatasets
using FreeConvection

for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

Nz = 32

training_datasets = tds = Dict(
    1 => NCDstack(datadep"lesbrary_free_convection_1/statistics.nc"),
    2 => NCDstack(datadep"lesbrary_free_convection_2/statistics.nc")
)

coarse_training_datasets = ctds = Dict(id => coarse_grain(ds, Nz) for (id, ds) in tds)

for id in keys(td)
    T_filepath = "free_convection_T_$id"
    animate_variable(tds[id][:T], ctds[id][:T], xlabel="Temperature T (°C)", filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$id"
    animate_variable(tds[id][:wT], ctds[id][:wT], xlabel="Heat flux wT (m/s °C)", filepath=wT_filepath, frameskip=5)
end
