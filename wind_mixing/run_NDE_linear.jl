using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim
using LinearAlgebra
using CairoMakie


# bc = (; uw=-5f-4, vw=0, wT=0)


EXTRACTED_FILE_PATH = joinpath(pwd(), "extracted_training_output", "NDE_18sim_windcooling_windheating_18simBFGST0.8nogradnonlocal_divide1f5_gradient_smallNN_leakyrelu_layers_1_rate_2e-4_T0.8_2e-4_extracted.jld2")
OUTPUT_DIR = joinpath(pwd(), "new_nonlocal_NDE_BFGS")
timestep = 60
stop_time = 60*11530
times = range(0, step=timestep, stop=stop_time)

uws = -3f-4:-1f-4:-5f-4
wTs = 1f-8:1f-8:3f-8

for uw in uws
    @info uw
    bc = (uw=uw, vw=0, wT=0)
    solve_oceananigans_modified_pacanowski_philander_nn_nonlocal_linear(bc, EXTRACTED_FILE_PATH, OUTPUT_DIR; 
                                                            timestep, stop_time, convective_adjustment=false)
end

for wT in wTs
    @info wT
    bc = (uw=0, vw=0, wT=wT)
    solve_oceananigans_modified_pacanowski_philander_nn_nonlocal_linear(bc, EXTRACTED_FILE_PATH, OUTPUT_DIR; 
                                                            timestep, stop_time, convective_adjustment=false)
end

f = jldopen("Output/test_linear_uw0_vw0_wT3.0e-8/NN_oceananigans.jld2")

fig = Figure()
ax = fig[1,1] = Axis(fig)
lines!(ax, f["timeseries/T/$(length(times) - 1)"][1, 1, :], f["grid/zC"][2:end-1])
# lines!(ax, f["timeseries/T/1000"][1, 1, :], f["grid/zC"][2:end-1])

display(fig)

close(f)