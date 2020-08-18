using Printf

using NCDatasets
using Plots
using Flux

using ClimateSurrogates
using Oceananigans.Utils

include("free_convection_npde.jl")

function animate_variable(ds, var; grid_points, xlabel, xlim, frameskip, fps)
    Nz, Nt = size(ds[var])

    z_fine = ds["zC"]
    z_coarse = coarse_grain(ds["zC"], grid_points)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $var [$n/$Nt]..."
        var_fine = ds[var][:, n]
        var_coarse = coarse_grain(ds[var][:, n], grid_points)

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(var_fine, z_fine, linewidth=2, xlim=xlim, ylim=(-100, 0),
             label="fine (Nz=$Nz)", xlabel=xlabel, ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(var_coarse, z_coarse, linewidth=2, label="coarse (Nz=$grid_points)")
    end

    filename = "free_convection_$var.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

function free_convection_heat_flux_training_data(ds; grid_points)
    T, wT = ds["T"], ds["wT"]
    Nz, Nt = size(T)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    training_data =
        [(coarse_grain( T[:, n], grid_points),
          coarse_grain(wT[:, n], grid_points))
         for n in 1:Nt]

    return training_data
end

function free_convection_time_step_training_data(ds; grid_points, future_time_steps=1)
    T = ds["T"]
    Nz, Nt = size(T)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    if future_time_steps == 1
        training_data =
            [(coarse_grain(T[:, n],   grid_points),
              coarse_grain(T[:, n+1], grid_points))
             for n in 1:Nt-1]
    else
        training_data =
            [(coarse_grain(T[:, n], grid_points),
              cat((coarse_grain(T[:, k], grid_points) for k in n+1:n+future_time_steps)..., dims=2))
             for n in 1:Nt-future_time_steps]
    end

    return training_data
end

function train_on_heat_flux!(NN, training_data, optimizer; epochs=1)
    loss(T, wT) = Flux.mse(NN(T), wT)

    function cb()
        Σloss = [loss(T, wT) for (T, wT) in training_data] |> sum
        @info "Training on heat flux... Σloss = $Σloss"
        return loss
    end

    for e in 1:epochs
        @info "Training on heat flux [epoch $e/$epochs]..."
        Flux.train!(loss, Flux.params(NN), training_data, optimizer, cb=cb)

        # # training_loss is declared local so it will be available for logging outside the gradient calculation.
        # local training_loss
        # ps = Flux.Params(NN)
        # for d in training_data
        #     gs = gradient(ps) do
        #         training_loss = loss(d...)
        #         # Code inserted here will be differentiated, unless you need that gradient information
        #         # it is better to do the work outside this block.
        #         return training_loss
        #     end
        #     # Insert whatever code you want here that needs training_loss, e.g. logging.
        #     # logging_callback(training_loss)
        #     # Insert what ever code you want here that needs gradient.
        #     # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
        #     Flux.update!(optimizer, ps, gs)
        #     # Here you might like to check validation set accuracy, and break out to do early stopping.
        #     cb()
        # end
    end

    return nothing
end

ds = NCDataset("free_convection_horizontal_averages.nc")

# Should not have saved constant units as strings...
nc_constant(ds, attr) = parse(Float64, ds.attrib[attr] |> split |> first)

Q = nc_constant(ds, "Heat flux")
ρ₀ = nc_constant(ds, "Reference density")
cₚ = nc_constant(ds, "Specific_heat_capacity")

surface_heat_flux = Q / (ρ₀ * cₚ)

Nz = 16  # Number of grid points for the neural PDE.

future_time_steps = 1

# animate_variable(ds, "T", grid_points=16, xlabel="Temperature T (°C)", xlim=(19, 20), frameskip=5, fps=15)
# animate_variable(ds, "wT", grid_points=16, xlabel="Heat flux", xlim=(-1e-5, 3e-5), frameskip=5, fps=15)

training_data_heat_flux = free_convection_heat_flux_training_data(ds, grid_points=Nz)
training_data_time_step = free_convection_time_step_training_data(ds, grid_points=Nz, future_time_steps=future_time_steps)

NN = free_convection_neural_pde_architecture(Nz, top_flux=surface_heat_flux, bottom_flux=0.0)

# train_on_heat_flux!(NN, training_data_heat_flux, Descent(1e-2), epochs=1)
