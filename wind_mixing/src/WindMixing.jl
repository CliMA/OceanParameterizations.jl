module WindMixing

export data, read_les_output,
       animate_prediction,
       mse, train_NDE, train_NDE_convective_adjustment, train_NN, animate_NN

using Flux, Plots
using Oceananigans.Grids: Cell, Face
using OceanParameterizations

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])
mse(x::Tuple{Array{Float32,2}, Array{Float64,2}}) = Flux.mse(Float64.(x[1]), x[2])

include("lesbrary_data.jl")
include("data_containers.jl")
include("NDE_training.jl")
include("NN_training.jl")
include("animation.jl")

end
