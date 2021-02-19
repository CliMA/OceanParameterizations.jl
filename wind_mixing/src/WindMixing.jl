module WindMixing

export data, read_les_output,
       animate_prediction,
       mse

using Flux, Plots
using Oceananigans.Grids: Center, Face
using OceanParameterizations

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])
mse(x::Tuple{Array{Float32,2}, Array{Float64,2}}) = Flux.mse(Float64.(x[1]), x[2])

include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")

end
