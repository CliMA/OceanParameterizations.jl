using Statistics
using NCDatasets
using DifferentialEquations
using Plots
using Flux, DiffEqFlux
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

u = Array(ds["u"])
v = Array(ds["v"])
T = Array(ds["T"])
uw = Array(ds["uw"])
vw = Array(ds["vw"])
wT = Array(ds["wT"])
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

# abstract type AbstractFeatureScaling end

# struct ZeroMeanUnitVarianceScaling{T} <: AbstractFeatureScaling
#     μ :: T
#     σ :: T
# end


# function ZeroMeanUnitVarianceScaling(data)
#     μ, σ = mean(data), std(data)
#     return ZeroMeanUnitVarianceScaling(μ, σ)
# end

# scale(x, s::ZeroMeanUnitVarianceScaling) = (x .- s.μ) / s.σ
# unscale(y, s::ZeroMeanUnitVarianceScaling) = s.σ * y .+ s.μ

u = Array(ds["u"])
u_scaling = ZeroMeanUnitVarianceScaling(u)

struct ScaledData
    scaled_data
    scaling
end

a = ScaledData(u_scaled, u_scaling)


# central derivative as gradient approximator, periodic boundary conditions

function central_difference(input, z)
    Δ = z[2] - z[1]
    output = similar(input)
    vals = @view output[2:length(output)-1]
    vals .= (@view(input[3:end]) .- @view(input[1:end-2])) ./ (2Δ)
    # output[1] = (input[2] - input[end]) / (2Δ)
    # output[end] = (input[1] - input[end-1])/(2Δ)
    output[1] = 0
    output[end] = 0
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

# function NDE!(dx, x, p, t)
#     f = p[1]
#     Nz = Int(p[2])
#     u = x[1:Nz]
#     v = x[Nz+1:2*Nz]
#     T = x[2*Nz+1:end]
#     dx[1:Nz] .= -1 .* central_difference(uw_NN(x), zC_coarse) .+ f .* v
#     dx[Nz+1:2*Nz] .= -1 .* central_difference(vw_NN(x), zC_coarse) .- f .* u
#     dx[2*Nz+1:end] .= -1 .* central_difference(face_to_cell(wT_NN(x)), zC_coarse)
# end

function cell_to_cell_derivative(D, data)
    face_data = D * data
    cell_data = 0.5 .* (@view(face_data[1:end-1]) .+ @view(face_data[2:end]))
    return cell_data
end

D_face = Dᶠ(Nz, 1/Nz)
cell_to_cell_derivative(D_face, u_scaled[:,20])

f = 10e-4
H = abs(zF_coarse[end] - zF_coarse[1])
τ = abs(t[end] - t[1])
Nz = length(zC_coarse)
u_scaling = ZeroMeanUnitVarianceScaling(u)
v_scaling = ZeroMeanUnitVarianceScaling(v)
T_scaling = ZeroMeanUnitVarianceScaling(T)
uw_scaling = ZeroMeanUnitVarianceScaling(uw)
vw_scaling = ZeroMeanUnitVarianceScaling(vw)
wT_scaling = ZeroMeanUnitVarianceScaling(wT)

p_nondimensional = (f, τ, H, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling)

function NDE_nondimensional!(dx, x, p, t)
    # uw_NN = reconstruct(weights, NN)
    f, τ, H, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling = p
    μ_u = u_scaling.μ
    μ_v = v_scaling.μ
    σ_u = u_scaling.σ
    σ_v = v_scaling.σ
    σ_T = T_scaling.σ
    σ_uw = uw_scaling.σ
    σ_vw = vw_scaling.σ
    σ_wT = wT_scaling.σ
    A = - τ / H
    B = f * τ
    D_face = Dᶠ(Nz, 1/Nz)
    D_cell = Dᶜ(Nz, 1/Nz)
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= A .* σ_uw ./ σ_u .* cell_to_cell_derivative(D_face, uw_NN(x)) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimentional gradient
    dx[Nz+1:2*Nz] .= A .* σ_vw ./ σ_v .* cell_to_cell_derivative(D_face, vw_NN(x)) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
    dx[2*Nz+1:end] .= A .* σ_wT ./ σ_T .* D_cell * wT_NN(x)
end

t_train, uvT_train = time_window(t, uvT_scaled, 1000)
# t_train, uvT_train = time_window(t, uvT_scaled, 100)

prob = ODEProblem(NDE_nondimensional!, uvT₀, (t_train[1]/τ, t_train[end]/τ), p_nondimensional, saveat=t_train./τ)

tpoint = 1000
sol = solve(prob)
plot(sol[:,tpoint][33:64], zC_coarse)
plot!(uvT_scaled[:,tpoint][33:64], zC_coarse)

function loss_NDE(x, y)
    return Flux.mse(x, y)
end

function cb()
    @info Flux.mse(Array(solve(prob)), uvT_train)
end

Flux.train!(loss_NDE, Flux.params([uw_NN, vw_NN, wT_NN]), zip(Array(solve(prob)),uvT_train), Descent(), cb = cb)

Flux.params(t_train, [2, 0])