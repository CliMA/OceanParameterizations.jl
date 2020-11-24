using Statistics
using NCDatasets
using DifferentialEquations
using Plots
using Flux, DiffEqFlux, Optim
using ClimateParameterizations
using Oceananigans.Grids
using BSON

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

D_cell = Dᶜ(Nz, zC_coarse[2] - zC_coarse[1])
D_face = Dᶠ(Nz, zF_coarse[2] - zF_coarse[1])

D_cell * uw_NN(uvT₀)
D_face * wT_NN(uvT₀)


uw_NN(uvT₀)

# uw_NN = Chain(Dense(32, 128, relu), Dense(128, 32))
prob_uw = NeuralODE(uw_NN, tspan_train, Tsit5(), saveat=t[1:100])
params(uw_NN)

Array(prob_uw(uw_scaled[:,1]))
prob_uw.p

zF = Array(ds["zF"])
diff(zF)

zF_linear_interpolation = coarse_grain_linear_interpolation(zF, 31, Face)

diff(zF_linear_interpolation)

function NDE!(dx, x, p, t)
    f = p[1]
    Nz = Int(p[2])
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] = - d_cell * uw_NN(x) + f*v
    dx[Nz+1:2*Nz] = - d_cell * vw_NN(x) + f*u
    dx[2*Nz+1:end] = - d_face * wT_NN(x)
end

prob = ODEProblem(NDE!, uvT₀, (0.,10.), [10e-4, 32])
solve(prob)

∂z(u) + fv