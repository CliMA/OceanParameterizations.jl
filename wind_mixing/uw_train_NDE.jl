using Statistics
using NCDatasets
using Plots
using Flux, DiffEqFlux, Optim
using ClimateSurrogates
using Oceananigans.Grids
using BSON

uw_NN_params = BSON.load("wind_mixing/Output/uw_NN_params.bson")

Nz = uw_NN_params[:grid_points]
NN = uw_NN_params[:neural_network]
u_scaled = uw_NN_params[:u_scaling]
v_scaled = uw_NN_params[:v_scaling]
T_scaled = uw_NN_params[:T_scaling]
uw_scaled = uw_NN_params[:uw_scaling]

PATH = joinpath(pwd(), "wind_mixing")
DATA_PATH = joinpath(PATH, "Data", "wind_mixing_horizontal_averages_0.02Nm2_8days.nc")

ds = NCDataset(DATA_PATH)

xC = Array(ds["xC"])
xF = Array(ds["xF"])
yC = Array(ds["yC"])
yF = Array(ds["yF"])
zC = Array(ds["zC"])
zF = Array(ds["zF"])

uT = Array(ds["uT"])
vT = Array(ds["vT"])
wT = Array(ds["wT"])

uu = Array(ds["uu"])
vv = Array(ds["vv"])
ww = Array(ds["ww"])
uv = Array(ds["uv"])
uw = Array(ds["uw"])
vw = Array(ds["vw"])

u = Array(ds["u"])
v = Array(ds["v"])

T = Array(ds["T"])
t = Array(ds["time"])

