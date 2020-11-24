using Statistics
using NCDatasets
using DifferentialEquations
using Plots
using Flux, DiffEqFlux, Optim
using ClimateParameterizations
using Oceananigans.Grids
using BSON

# load dataset and NN
PATH = joinpath(pwd(), "wind_mixing")
DATA_PATH = joinpath(PATH, "Data", "wind_mixing_horizontal_averages_0.02Nm2_8days.nc")
ds = NCDataset(DATA_PATH)
@info ds.attrib
t = Array(ds["time"])

uw_NN_params = BSON.load(joinpath(PATH, "Output","uw_NN_params.bson"))
vw_NN_params = BSON.load(joinpath(PATH, "Output","vw_NN_params.bson"))
wT_NN_params = BSON.load(joinpath(PATH, "Output","wT_NN_params.bson"))

uw_NN = uw_NN_params[:neural_network]
vw_NN = vw_NN_params[:neural_network]
wT_NN = wT_NN_params[:neural_network]

Nz = uw_NN_params[:grid_points]
u_scaled = uw_NN_params[:u_scaling]
v_scaled = uw_NN_params[:v_scaling]
T_scaled = uw_NN_params[:T_scaling]
uw_scaled = uw_NN_params[:uw_scaling]
vw_scaled = vw_NN_params[:vw_scaling]
wT_scaled = wT_NN_params[:wT_scaling]
uvT_scaled = uw_NN_params[:uvT_scaling]
zC_coarse = uw_NN_params[:zC]
zF_coarse = wT_NN_params[:zF]

tspan_train = (0.0, t[100])
uvT₀ = uvT_scaled[:,1]

Nz = 32

# central derivative as gradient approximator, periodic boundary conditions

function central_difference(input, z)
    Δ = z[2] - z[1]
    output = similar(input)
    vals = @view output[2:length(output)-1]
    vals .= (@view(input[3:end]) .-@view(input[1:end-2])) ./ (2Δ)
    output[1] = (input[2] - input[end]) / (2Δ)
    output[end] = (input[1] - input[end-1])/(2Δ)
    return output
end

# interpolation from face centered values to cell centered values
function face_to_cell(input)
    output = similar(input, length(input)-1)
    output .= (@view(input[1:end-1]) .+ @view(input[2:end]) ) ./ 2
    return output
end

# splicing data to train the NN
function time_window(t, uvT, stopindex)
    if stopindex < length(t)
        return (t[1:stopindex], uvT[:,1:stopindex])
    else
        @info "stop index larger than length of t"
    end
end

function NDE!(dx, x, p, t)
    f = p[1]
    Nz = Int(p[2])
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= -1 .* central_difference(uw_NN(x), zC_coarse) .+ f .* v
    dx[Nz+1:2*Nz] .= -1 .* central_difference(vw_NN(x), zC_coarse) .- f .* u
    dx[2*Nz+1:end] .= -1 .* central_difference(face_to_cell(wT_NN(x)), zC_coarse)
end

t_train, uvT_train = time_window(t, uvT_scaled, 2)

prob = ODEProblem(NDE!, uvT₀, (t_train[1],t_train[end]), [10e-4, 32], saveat=t_train)

solve(prob)

function loss_NDE(x, y)
    return Flux.mse(x, y)
end

Flux.train!(loss_NDE, Flux.params([uw_NN, vw_NN, wT_NN]), zip(Array(solve(prob)),uvT_train), ADAM())

