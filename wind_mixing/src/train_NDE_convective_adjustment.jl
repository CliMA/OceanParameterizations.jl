using Statistics
using NCDatasets
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using GalacticOptim
using WindMixing

function time_window(t, uvT, trange)
    return (Float32.(t[trange]), Float32.(uvT[:,trange]))
end
