module DataWrangling

export
       # Coarse graining
       coarse_grain,

       # Feature scaling
       AbstractFeatureScaling, ZeroMeanUnitVarianceScaling, MinMaxScaling, scale, unscale,

       # Convective adjustment
       convective_adjust!,

       # Animations
       animate_gif

using Statistics
using OrderedCollections
using OceanTurb
using Plots

include("coarse_graining.jl")
include("feature_scaling.jl")
include("convective_adjust.jl")
include("animate_gif.jl")

end
