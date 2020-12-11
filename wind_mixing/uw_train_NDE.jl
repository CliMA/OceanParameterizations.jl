using OceanParameterizations
using Flux, OceanTurb, Plots
using DifferentialEquations
using Oceananigans.Grids: Cell, Face

include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")
output_gif_directory = "Output1"

##

train_files = ["strong_wind", "free_convection"]
test_file = "strong_wind"

##

𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")
𝒟test = data(test_file,
                    override_scalings=𝒟train.scalings, # use the scalings from the training data
                    animate=false,
                    animate_dir="$(output_gif_directory)/Testing")

les = read_les_output(test_file)

## Neural Networks

# trained NN models
uw_NN_model = nn_model(𝒱 = 𝒟train.uw,
                    model = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,33)),
                    optimizers = [ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent()],
                   )

vw_NN_model = nn_model(𝒱 = 𝒟train.vw,
                    model = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,33)),
                    optimizers = [ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent()],
                   )

wT_NN_model = nn_model(𝒱 = 𝒟train.wT,
                    model = Chain(Dense(96,96, relu), Dense(96,96, relu), Dense(96,33)),
                    optimizers = [ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent()],
                   )

uw_NN = predict(𝒟test.uw, uw_NN_model)
vw_NN = predict(𝒟test.vw, vw_NN_model)
wT_NN = predict(𝒟test.wT, wT_NN_model)

# Compare NN predictions to truth
animate_prediction(uw_NN, "uw", 𝒟test, test_file; legend_labels=["NN(u,v,T)", "truth"], filename="uw_NN_$(test_file)", directory=output_gif_directory)
animate_prediction(vw_NN, "vw", 𝒟test, test_file; legend_labels=["NN(u,v,T)", "truth"], filename="vw_NN_$(test_file)", directory=output_gif_directory)
animate_prediction(wT_NN, "wT", 𝒟test, test_file; legend_labels=["NN(u,v,T)", "truth"], filename="wT_NN_$(test_file)", directory=output_gif_directory)

## Gaussian Process Regression

# A. Find the kernel that minimizes the prediction error on the training data
# * Sweeps over length-scale hyperparameter value in logγ_range
# * Sweeps over covariance functions
logγ_range=-1.0:0.5:1.0 # sweep over length-scale hyperparameter
# uncomment the next three lines to try this but just for testing the GPR use the basic get_kernel stuff below
# uw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
# vw_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)
# wT_kernel = best_kernel(𝒟train.uw, logγ_range=logγ_range)

# OR set the kernel manually here (to save a bunch of time):
uw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
vw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
wT_kernel = get_kernel(1,0.1,0.0,euclidean_distance)

# Trained GP models
uw_GP_model = gp_model(𝒟train.uw, uw_kernel)
vw_GP_model = gp_model(𝒟train.vw, vw_kernel)
wT_GP_model = gp_model(𝒟train.wT, wT_kernel)

# GP predictions on test data
uw_GP = predict(𝒟test.uw, uw_GP_model)
vw_GP = predict(𝒟test.vw, vw_GP_model)
wT_GP = predict(𝒟test.wT, wT_GP_model)

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])
mse(uw_GP)
mse(vw_GP)
mse(wT_GP)

# Compare GP predictions to truth
animate_prediction(uw_GP, "uw", 𝒟test, test_file; legend_labels=["GP(u,v,T)", "truth"], filename="uw_GP_$(test_file)", directory=output_gif_directory)
animate_prediction(vw_GP, "vw", 𝒟test, test_file; legend_labels=["GP(u,v,T)", "truth"], filename="vw_GP_$(test_file)", directory=output_gif_directory)
animate_prediction(wT_GP, "wT", 𝒟test, test_file; legend_labels=["GP(u,v,T)", "truth"], filename="wT_GP_$(test_file)", directory=output_gif_directory)

## KPP Parameterization (no training)

# les = read_les_output(test_file)
parameters = KPP.Parameters() # default parameters
KPP_model = closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)
predictions = KPP_model()
T_KPP = (predictions, 𝒟test.T.coarse)
mse(T_KPP)
animate_prediction(T_KPP, "T", 𝒟test, test_file; legend_labels=["KPP(T)", "truth"], filename="T_KPP_$(test_file)", directory=output_gif_directory)

## TKE Parameterization (no training; use default parameters)

# les = read_les_output(test_file)
parameters = TKEMassFlux.TKEParameters() # default parameters
TKE_model = closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], Δt, les)()
predictions = TKE_model()
T_TKE = (predictions, 𝒟test.T.coarse)
mse(T_TKE)
animate_prediction(T_TKE, "T", 𝒟test, test_file; legend_labels=["TKE(T)", "truth"], filename="T_TKE_$(test_file)", directory=output_gif_directory)

## Solving the PDEs using the predictions from NN or GP models

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

function cell_to_cell_derivative(D, data)
    face_data = D * data
    cell_data = 0.5 .* (@view(face_data[1:end-1]) .+ @view(face_data[2:end]))
    return cell_data
end

f = les.fᶿ
zF_coarse = 𝒟test.uw.z
zC_coarse = 𝒟test.u.z
H = abs(zF_coarse[end] - zF_coarse[1])
τ = abs(𝒟test.t[end] - 𝒟test.t[1])
Nz = length(zC_coarse)
uvT_scaled = 𝒟test.uvT_scaled
tspan_train = (0.0, t[2])
uvT₀ = uvT_scaled[:,1]

get_μ_σ(name) = (𝒟test.scalings[name].μ, 𝒟test.scalings[name].σ)
μ_u, σ_u = get_μ_σ("u")
μ_v, σ_v = get_μ_σ("v")
μ_T, σ_T = get_μ_σ("T")
μ_uw, σ_uw = get_μ_σ("uw")
μ_vw, σ_vw = get_μ_σ("vw")
μ_wT, σ_wT = get_μ_σ("wT")

uw_weights, re_uw = Flux.destructure(uw_NN_model)
vw_weights, re_vw = Flux.destructure(vw_NN_model)
wT_weights, re_wT = Flux.destructure(wT_NN_model)

p_nondimensional = Float32.(cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1))

function NDE_nondimensional!(dx, x, p, t)
    f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT = p[1:12]
    Nz = 32
    uw_weights = p[13:21740]
    vw_weights = p[21741:43468]
    wT_weights = p[43469:end]
    uw_NN_model = re_uw(uw_weights)
    vw_NN_model = re_vw(vw_weights)
    wT_NN_model = re_wT(wT_weights)
    A = - τ / H
    B = f * τ
    D_face = Dᶠ(Nz, 1/Nz)
    D_cell = Dᶜ(Nz, 1/Nz)
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= A .* σ_uw ./ σ_u .* cell_to_cell_derivative(D_face, uw_NN_model(x)) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
    dx[Nz+1:2*Nz] .= A .* σ_vw ./ σ_v .* cell_to_cell_derivative(D_face, vw_NN_model(x)) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
    dx[2*Nz+1:end] .= A .* σ_wT ./ σ_T .* (D_cell * wT_NN_model(x))
end

t_train, uvT_train = time_window(𝒟test.t, 𝒟test.uvT_scaled, 10)
t_train = Float32.(t_train ./ τ)
# t_train, uvT_train = time_window(t, uvT_scaled, 100)
prob = ODEProblem(NDE_nondimensional!, uvT₀, (t_train[1], t_train[end]), p_nondimensional, saveat=t_train) # divide τ needs to be changed

# tpoint = 1000
sol = solve(prob)
# plot(sol[:,tpoint][33:64], zC_coarse)
# plot!(uvT_scaled[:,tpoint][33:64], zC_coarse)

function loss_NDE_NN()
    p = Float32.(cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1))
    # _prob = remake(prob, p=p)
    _sol = Array(solve(prob, ROCK4(), p=p, reltol=1e-3, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    loss = Flux.mse(_sol, uvT_train)
    return loss
end

function cb()
    p = cat(f, τ, H, Nz, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, uw_weights, vw_weights, wT_weights, dims=1)
    # _prob = remake(prob, p=p)
    _sol = Array(solve(prob, ROCK4(), p=p, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    loss = Flux.mse(_sol, uvT_train)
    @info loss
    return _sol
end

Flux.train!(loss_NDE_NN, Flux.params(uw_weights, vw_weights, wT_weights), Iterators.repeated((), 100), ADAM(0.01), cb=Flux.throttle(cb, 2))

tpoint = 100
_sol = cb()
plot(_sol[:,tpoint][33:64], zC_coarse, label="NDE")
plot!(uvT_scaled[:,tpoint][33:64], zC_coarse, label="truth")
plot(_sol[:,tpoint][1:32], zC_coarse, label="NDE")
plot!(uvT_scaled[:,tpoint][1:32], zC_coarse, label="truth")
plot(_sol[:,tpoint][65:end], zC_coarse, label="NDE", legend=:topleft)
plot!(uvT_scaled[:,tpoint][65:end], zC_coarse, label="truth")
