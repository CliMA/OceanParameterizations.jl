using Statistics
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using LaTeXStrings
using WindMixing

# data in which the neural network is trained on
train_files = ["strong_wind"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "Output")

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

# produce a gif animation to visualize the profiles as given by NN and the simulations
function animate_NN(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str, PATH=joinpath(pwd(), "Output"))
    anim = @animate for n in 1:size(xs[1], 2)
        x_max = maximum(maximum(x) for x in xs)
        x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1], 2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n] / 86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
end

# Prepare dataset into  tuple form for Flux.train!
function prepare_training_data(input, truth)
    return [(input[:,i], truth[:,i]) for i in 1:size(truth, 2)]
end

uw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.uw.scaled)
vw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.vw.scaled)
wT_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.wT.scaled)

N_inputs = 96
N_outputs = 31
uw_NN_model = Chain(Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_outputs))
vw_NN_model = Chain(Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_outputs))
wT_NN_model = Chain(Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_inputs, relu), Dense(N_inputs, N_outputs))

function predict(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

loss_uw(x, y) = Flux.Losses.mse(predict(uw_NN_model, x, y), y)
loss_vw(x, y) = Flux.Losses.mse(predict(vw_NN_model, x, y), y)
loss_wT(x, y) = Flux.Losses.mse(predict(wT_NN_model, x, y), y)

uw_loss = []
vw_loss = []
wT_loss = []

# training the neural networks and records the loss to produce a learning curve
function train_NN_learning_curve(NN, loss, data, opts, loss_list)
    function cb()
        push!(loss_list, mean([loss(data[i][1], data[i][2]) for i in 1:length(data)]))
    end
    for opt in opts
        @info "loss = $(mean([loss(data[i][1], data[i][2]) for i in 1:length(data)]))"
        Flux.train!(loss, params(NN), data, opt, cb=cb)
    end 
end

optimizers = [ADAM(0.01), ADAM(0.01), ADAM(0.01), Descent(), Descent(), Descent(), Descent(),  Descent(),  Descent()]

train_NN_learning_curve(uw_NN_model, loss_uw, uw_train, optimizers, uw_loss)
train_NN_learning_curve(vw_NN_model, loss_vw, vw_train, optimizers, vw_loss)
train_NN_learning_curve(wT_NN_model, loss_wT, wT_train, optimizers, wT_loss)

@info "uw loss = $(mean([loss_uw(uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train)]))"
@info "vw loss = $(mean([loss_vw(vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train)]))"
@info "wT loss = $(mean([loss_wT(wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train)]))"

l = @layout [a b c]
p1 = plot(1:length(uw_loss), uw_loss, label=nothing, leftmargin=5Plots.mm, bottommargin=5Plots.mm)
xlabel!(p1, L"Iterations")
ylabel!(p1, L"$Loss(\mathbb{NN}_1, \overline{U'W'})$")
title!(p1, L"\mathbb{NN}_1")

p2 = plot(1:length(vw_loss), vw_loss,  label=nothing)
xlabel!(p2, L"Iterations")
ylabel!(p2, L"$Loss(\mathbb{NN}_2, \overline{V'W'})$")
title!(p2, L"\mathbb{NN}_2")

p3 = plot(1:length(wT_loss), wT_loss,  label=nothing)
xlabel!(p3, L"Iterations")
ylabel!(p3, L"$Loss(\mathbb{NN}_3, \overline{W'T'})$")
title!(p3, L"\mathbb{NN}_3")

fig = plot(p1, p2, p3, layout=l, size=(1700, 600))

# saves the trained neural network
uw_NN_params = Dict(
    :neural_network => uw_NN_model)

bson(joinpath(pwd(), "NDEs", "uw_NN.bson"), uw_NN_params)

vw_NN_params = Dict(
    :neural_network => vw_NN_model)

bson(joinpath(pwd(), "NDEs", "vw_NN.bson"), vw_NN_params)

wT_NN_params = Dict(
    :neural_network => wT_NN_model)

bson(joinpath(pwd(), "NDEs", "wT_NN.bson"), wT_NN_params)

# plots the profiles at different frames
NN_prediction_uw = cat((predict(uw_NN_model, uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train))..., dims=2)
truth_uw = cat((uw_train[i][2] for i in 1:length(uw_train))..., dims=2)
uw_plots = (NN_prediction_uw, truth_uw)
index₁ = 5
index₂ = 280

l = @layout [a b]
p1 = plot(uw_plots[1][:,index₁], 𝒟train.uw.z, label=L"\mathbb{NN}_1 (U, V, T)", legend=:bottomleft)
plot!(p1, uw_plots[2][:,index₁], 𝒟train.uw.z, label=L"Truth")
title!(p1, "Timestep $index₁")
p2 = plot(uw_plots[1][:,index₂], 𝒟train.uw.z, label=L"\mathbb{NN}_1 (U, V, T)", legend=:bottomleft)
plot!(p2, uw_plots[2][:,index₂], 𝒟train.uw.z, label=L"Truth")
title!(p2, "Timestep $index₂")
fig = plot(p1, p2, layout=l)
ylabel!(fig, L"z/m")
xlabel!(fig, L"\overline{U'W'}")
display(fig)

NN_prediction_vw = cat((predict(vw_NN_model, vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train))..., dims=2)
truth_vw = cat((vw_train[i][2] for i in 1:length(vw_train))..., dims=2)
vw_plots = (NN_prediction_vw, truth_vw)

l = @layout [a b]
p1 = plot(vw_plots[1][:,index₁], 𝒟train.uw.z, label=L"\mathbb{NN}_2 (U, V, T)", legend=:bottomleft)
plot!(p1, vw_plots[2][:,index₁], 𝒟train.uw.z, label=L"Truth")
title!(p1, "Timestep $index₁")
p2 = plot(vw_plots[1][:,index₂], 𝒟train.uw.z, label=L"\mathbb{NN}_2 (U, V, T)", legend=:bottomleft)
plot!(p2, vw_plots[2][:,index₂], 𝒟train.uw.z, label=L"Truth")
title!(p2, "Timestep $index₂")
fig = plot(p1, p2, layout=l)
ylabel!(fig, L"z/m")
xlabel!(fig, L"\overline{V'W'}")
display(fig)

NN_prediction_wT = cat((predict(wT_NN_model, wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train))..., dims=2)
truth_wT = cat((wT_train[i][2] for i in 1:length(wT_train))..., dims=2)
wT_plots = (NN_prediction_wT, truth_wT)

l = @layout [a b]
p1 = plot(wT_plots[1][:,index₁], 𝒟train.uw.z, label=L"\mathbb{NN}_3 (U, V, T)", legend=:bottomleft)
plot!(p1, wT_plots[2][:,index₁], 𝒟train.uw.z, label=L"Truth")
title!(p1, "Timestep $index₁")
p2 = plot(wT_plots[1][:,index₂], 𝒟train.uw.z, label=L"\mathbb{NN}_3 (U, V, T)", legend=:bottomleft)
plot!(p2, wT_plots[2][:,index₂], 𝒟train.uw.z, label=L"Truth")
title!(p2, "Timestep $index₂")
fig = plot(p1, p2, layout=l)
ylabel!(fig, L"z/m")
xlabel!(fig, L"\overline{W'T'}")
display(fig)

# animate the profiles
animate_NN(uw_plots, 𝒟train.uw.z, 𝒟train.t[:,1], "uw", ["NN", "truth"], "uw_SWNH")
animate_NN(vw_plots, 𝒟train.vw.z, 𝒟train.t[:,1], "vw", ["NN", "truth"], "vw_SWNH")
animate_NN(wT_plots, 𝒟train.wT.z, 𝒟train.t[:,1], "wT", ["NN", "truth"], "wT_SWNH")