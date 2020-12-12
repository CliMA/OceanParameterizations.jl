using OceanParameterizations
using WindMixing
using Flux

## Pick training and test simulations

output_gif_directory = "Output1"

train_files = ["strong_wind", "free_convection"]
test_file = "strong_wind"

𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")
𝒟test = data(test_file,
                    override_scalings=𝒟train.scalings, # use the scalings from the training data
                    animate=false,
                    animate_dir="$(output_gif_directory)/Testing")
les = read_les_output(test_file)

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

t = 𝒟test.t
Nz = 32
uvT₀ = 𝒟test.uvT_scaled[:,1]
zF_coarse = 𝒟test.uw.z
f⁰ = les.f⁰

∂z(vec) = (vec[1:Nz] .- vec[2:Nz+1]) ./ diff(zF_coarse)
function f(dx, x, p, t)
    u = x[1:Nz]
    v = x[Nz+1:2*Nz]
    T = x[2*Nz+1:end]
    dx[1:Nz] .= -∂z(uw_GP_model(x)) .+ f⁰ .* v
    dx[Nz+1:2*Nz] .= -∂z(vw_GP_model(x)) .- f⁰ .* u
    dx[2*Nz+1:end] .= -∂z(wT_GP_model(x))
end

prob = ODEProblem(f, uvT₀, (t[1],t[288]), 1, saveat=t)
sol = solve(prob, ROCK4())

tpoint = 288
_sol = cb()
plot(_sol[:,tpoint][33:64], zC_coarse, label="NDE")
plot!(uvT_scaled[:,tpoint][33:64], zC_coarse, label="truth")
plot(_sol[:,tpoint][1:32], zC_coarse, label="NDE")
plot!(uvT_scaled[:,tpoint][1:32], zC_coarse, label="truth")
plot(_sol[:,tpoint][65:end], zC_coarse, label="NDE", legend=:topleft)
plot!(uvT_scaled[:,tpoint][65:end], zC_coarse, label="truth")
