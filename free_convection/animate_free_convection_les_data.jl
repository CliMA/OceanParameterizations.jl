using NCDatasets
using OceanParameterizations
using FreeConvection

Qs = [25, 50, 75, 100]

ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs)

for Q in Qs
    T_filepath = "free_convection_T_$(Q)W.mp4"
    animate_variable(ds[Q], "T", grid_points=32, xlabel="Temperature T (°C)", xlim=(19, 20),
                     filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$(Q)W.mp4"
    animate_variable(ds[Q], "wT", grid_points=32, xlabel="Heat flux wT (m/s °C)", xlim=(-1e-5, 3e-5),
                     filepath=wT_filepath, frameskip=5)
end
