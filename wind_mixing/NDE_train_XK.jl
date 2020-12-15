using Statistics
using BSON
using OrdinaryDiffEq
using Flux
using NCDatasets
using Plots
using DiffEqSensitivity
using JLD2
using Oceananigans.Grids
using OceanParameterizations
# ENV["CUDA_VISIBLE_DEVICES"]="0"

# load dataset and NN
PATH = pwd()
DATA_PATH = joinpath(PATH, "Data", "wind_mixing_horizontal_averages_0.02Nm2_8days.nc")
ds = NCDataset(DATA_PATH)
@info ds.attrib
t = Array(ds["time"])
# 
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

tspan_train = (0.0, t[2])
uvT₀ = uvT_scaled[:,1]

Nz = 32


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
function time_window(t, uvT; startindex=1, stopindex)
    if stopindex < length(t)
        return (t[startindex:stopindex], uvT[:,startindex:stopindex])
    else
        @info "stop index larger than length of t"
    end
end

function cell_to_cell_derivative(D, data)
    face_data = D * data
    cell_data = 0.5 .* (@view(face_data[1:end-1]) .+ @view(face_data[2:end]))
    return cell_data
end

f = 1f-4
H = abs(zF_coarse[end] - zF_coarse[1])
τ = abs(t[end] - t[1])
Nz = length(zC_coarse)
u_scaling = ZeroMeanUnitVarianceScaling(u)
v_scaling = ZeroMeanUnitVarianceScaling(v)
T_scaling = ZeroMeanUnitVarianceScaling(T)
uw_scaling = ZeroMeanUnitVarianceScaling(uw)
vw_scaling = ZeroMeanUnitVarianceScaling(vw)
wT_scaling = ZeroMeanUnitVarianceScaling(wT)
μ_u = u_scaling.μ
μ_v = v_scaling.μ
σ_u = u_scaling.σ
σ_v = v_scaling.σ
σ_T = T_scaling.σ
σ_uw = uw_scaling.σ
σ_vw = vw_scaling.σ
σ_wT = wT_scaling.σ
uw_weights, re_uw = Flux.destructure(uw_NN)
vw_weights, re_vw = Flux.destructure(vw_NN)
wT_weights, re_wT = Flux.destructure(wT_NN)

p_nondimensional = Float32.(cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1))

function NDE_nondimensional!(dx, x, p, t)
    f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT = p[1:12]
    Nz = 32
    uw_weights = p[13:21740]
    vw_weights = p[21741:43468]
    wT_weights = p[43469:end]
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    A = - τ / H
    B = f * τ
    D_face = Dᶠ(Nz, 1/Nz)
    D_cell = Dᶜ(Nz, 1/Nz)
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= A .* σ_uw ./ σ_u .* cell_to_cell_derivative(D_face, uw_NN(x)) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
    dx[Nz+1:2*Nz] .= A .* σ_vw ./ σ_v .* cell_to_cell_derivative(D_face, vw_NN(x)) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
    dx[2*Nz+1:end] .= A .* σ_wT ./ σ_T .* (D_cell * wT_NN(x))
end


function NDE_nondimensional_flux!(dx, x, p, t)
    f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT = p[1:12]
    Nz = 32
    uw_weights = p[13:21740]
    vw_weights = p[21741:43468]
    wT_weights = p[43469:end]
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    A = - τ / H
    B = f * τ
    D_face = Dᶠ(Nz, 1/Nz)
    D_cell = Dᶜ(Nz, 1/Nz)
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= A .* σ_uw ./ σ_u .* cell_to_cell_derivative(D_face, uw_NN(x)) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
    dx[Nz+1:2*Nz] .= A .* σ_vw ./ σ_v .* cell_to_cell_derivative(D_face, vw_NN(x)) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
    dx[2*Nz+1:end] .= A .* σ_wT ./ σ_T .* (D_cell * wT_NN(x))
end


start=100
stop=150
t_train, uvT_train = time_window(t, uvT_scaled,startindex=start, stopindex=stop)
t_train = Float32.((t_train .- t[start]) ./ τ)
# t_train, uvT_train = time_window(t, uvT_scaled, 100)
uvT₀ = uvT_scaled[:,start]
prob = ODEProblem(NDE_nondimensional!, uvT₀, (0f0, t_train[end] - t_train[1]), p_nondimensional, saveat=t_train) # divide τ needs to be changed


tpoint = 150
sol = Array(solve(prob, ROCK4()))
plot(sol[:,tpoint-start+1][33:64], zC_coarse)
plot!(uvT_scaled[:,tpoint][33:64], zC_coarse)

opt = Tsit5()

function loss_NDE_NN()
    p = Float32.(cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1))
    # _prob = remake(prob, p=p)
    _sol = Array(solve(prob, opt, p=p, reltol=1e-3, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    loss = Flux.mse(_sol, uvT_train)
    return loss
end

function cb()
    p = cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1)
    # _prob = remake(prob, p=p)
    _sol = Array(solve(prob, opt, p=p, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    loss = Flux.mse(_sol, uvT_train)
    @info loss
    return _sol
end

Flux.train!(loss_NDE_NN, Flux.params(uw_weights, vw_weights, wT_weights), Iterators.repeated((), 100), ADAM(0.01), cb=Flux.throttle(cb, 2))

@info uvT₀

tpoint = 55
_sol = cb()
plot(_sol[:,tpoint-start+1][33:64], zC_coarse, label="NDE")
plot!(uvT_scaled[:,tpoint][33:64], zC_coarse, label="truth")
plot(_sol[:,tpoint-start+1][1:32], zC_coarse, label="NDE", legend=:bottomright)
plot!(uvT_scaled[:,tpoint][1:32], zC_coarse, label="truth")
xlabel!("uw")
ylabel!("z")
# savefig("Output/uw_zigzag.png")
plot(_sol[:,tpoint-start+1][65:end], zC_coarse, label="NDE", legend=:topleft)
plot!(uvT_scaled[:,tpoint][65:end], zC_coarse, label="truth")