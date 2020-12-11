using DataDeps
using GeoData
using NCDatasets
using FreeConvection

for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

ds = Dict(
    1 => NCDataset(datadep"lesbrary_free_convection_1/statistics.nc"),
    2 => NCDataset(datadep"lesbrary_free_convection_2/statistics.nc")
)

for id in keys(ds)
    T_filepath = "free_convection_T_$id.mp4"
    animate_variable(ds[id], "T", grid_points=32, xlabel="Temperature T (°C)", xlim=(19, 20),
                     filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$id.mp4"
    animate_variable(ds[id], "wT", grid_points=32, xlabel="Heat flux wT (m/s °C)", xlim=(-1e-5, 3e-5),
                     filepath=wT_filepath, frameskip=5)
end
