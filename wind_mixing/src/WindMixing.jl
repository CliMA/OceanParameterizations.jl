module WindMixing

export data, read_les_output,
       animate_prediction

using Flux, Plots
using Oceananigans.Grids: Cell, Face
using OceanParameterizations

include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")

end
