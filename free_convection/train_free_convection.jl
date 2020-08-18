using NCDatasets

using ClimateSurrogates

function free_convection_training_data(ds; grid_points, future_time_steps=1)
    T = ds["T"]
    Nz, Nt = size(T)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    if future_time_steps == 1
        training_data =
            [Pair(coarse_grain(T[:, n],   grid_points),
                  coarse_grain(T[:, n+1], grid_points))
             for n in 1:Nt-1]
    else
        training_data =
            [Pair(coarse_grain(T[:, n], grid_points),
                  cat((coarse_grain(T[:, k], grid_points) for k in n+1:n+future_time_steps)..., dims=2))
             for n in 1:Nt-future_time_steps]
    end

    return training_data
end

ds = NCDataset("free_convection_horizontal_averages.nc")

Nz = 16  # Number of grid points for the neural PDE.

training_data = free_convection_training_data(ds, grid_points=Nz, future_time_steps=1)
