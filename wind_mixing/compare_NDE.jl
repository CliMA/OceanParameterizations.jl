using Statistics
using NCDatasets
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")

# train_files = ["strong_wind", "strong_wind_weak_heating"]
PATH = pwd()

uw_NDE = BSON.load(joinpath(PATH, "Output", "uw_NDE_1sim_100.bson"))[:neural_network]
vw_NDE = BSON.load(joinpath(PATH, "Output", "vw_NDE_1sim_100.bson"))[:neural_network]
wT_NDE = BSON.load(joinpath(PATH, "Output", "wT_NDE_1sim_100.bson"))[:neural_network]

# uw_weights = BSON.load(joinpath(PATH, "Output", "uw_NDE_weights_2DaySuite.bson"))[:weights]
# vw_weights = BSON.load(joinpath(PATH, "Output", "vw_NDE_weights_2DaySuite.bson"))[:weights]
# wT_weights = BSON.load(joinpath(PATH, "Output", "wT_NDE_weights_2DaySuite.bson"))[:weights]


# uw_NN_1sim_100 = re_uw(uw_weights)
# vw_NN_1sim_100 = re_vw(vw_weights)
# wT_NN_1sim_100 = re_wT(wT_weights)

# uw_NDE = Dict(:neural_network => uw_NN_1sim_100)
# bson(joinpath(PATH, "Output", "uw_NDE_1sim_100.bson"), uw_NDE)
# vw_NDE = Dict(:neural_network => vw_NN_1sim_100)
# bson(joinpath(PATH, "Output", "vw_NDE_1sim_100.bson"), vw_NDE)
# wT_NDE = Dict(:neural_network => wT_NN_1sim_100)
# bson(joinpath(PATH, "Output", "wT_NDE_1sim_100.bson"), wT_NDE)

# uw_weights, re_uw = Flux.destructure(uw_NDE)
# vw_weights, re_vw = Flux.destructure(vw_NDE)
# wT_weights, re_wT = Flux.destructure(wT_NDE)

# uw_weights = BSON.load(joinpath(PATH, "Output", "uw_NDE_weights_2DaySuite.bson"))[:weights]
# vw_weights = BSON.load(joinpath(PATH, "Output", "vw_NDE_weights_2DaySuite.bson"))[:weights]
# wT_weights = BSON.load(joinpath(PATH, "Output", "wT_NDE_weights_2DaySuite.bson"))[:weights]
# uw_NDE = re_uw(uw_weights)
# vw_NDE = re_vw(vw_weights)
# wT_NDE = re_wT(wT_weights)


function test_NDE(𝒟train, uw_NDE, vw_NDE, wT_NDE, trange)
    test_files = ["strong_wind", "strong_wind_weak_heating", "strong_wind_weak_cooling", "strong_wind_no_coriolis", "free_convection", "weak_wind_strong_cooling"]
    output_gif_directory = "Output"
    PATH = pwd()

    # 𝒟 = data(test_files, scale_type=ZeroMeanUnitVarianceScaling, animate=false, animate_dir="$(output_gif_directory)/Training")
    𝒟tests = [data(test_file, scale_type=ZeroMeanUnitVarianceScaling, animate=false, animate_dir="$(output_gif_directory)/Training") for test_file in test_files]

    tsteps = size(𝒟train.t[:,1], 1)

    function predict_NDE(NN, x, top, bottom)
        interior = NN(x)
        return [top; interior; bottom]
    end

    H = Float32(abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1]))
    τ = Float32(abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1]))
    Nz = 𝒟train.grid_points - 1
    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]
    μ_u = Float32(u_scaling.μ)
    μ_v = Float32(v_scaling.μ)
    σ_u = Float32(u_scaling.σ)
    σ_v = Float32(v_scaling.σ)
    σ_T = Float32(T_scaling.σ)
    σ_uw = Float32(uw_scaling.σ)
    σ_vw = Float32(vw_scaling.σ)
    σ_wT = Float32(wT_scaling.σ)

    uw_tops = [Float32(𝒟test.uw.scaled[1,1]) for 𝒟test in 𝒟tests]

    uw_bottom₁ = Float32(uw_scaling(-1f-3))
    uw_bottom₂ = Float32(uw_scaling(-1f-3))
    uw_bottom₃ = Float32(uw_scaling(-8f-4))
    uw_bottom₄ = Float32(uw_scaling(-2f-4))
    uw_bottom₅ = Float32(uw_scaling(0f0))
    uw_bottom₆ = Float32(uw_scaling(-3f-4))
    uw_bottoms = [uw_bottom₁, uw_bottom₂, uw_bottom₃, uw_bottom₄, uw_bottom₅, uw_bottom₆]

    vw_tops = [Float32(𝒟test.vw.scaled[1,1]) for 𝒟test in 𝒟tests]

    vw_bottoms = [Float32(𝒟test.vw.scaled[end,1]) for 𝒟test in 𝒟tests]

    wT_tops = [Float32(𝒟test.wT.scaled[1,1]) for 𝒟test in 𝒟tests]

    wT_bottom₁ = Float32(wT_scaling(0f0))
    wT_bottom₂ = Float32(wT_scaling(-4f-8))
    wT_bottom₃ = Float32(wT_scaling(3f-8))
    wT_bottom₄ = Float32(wT_scaling(0f0))
    wT_bottom₅ = Float32(wT_scaling(1.2f-7))
    wT_bottom₆ = Float32(wT_scaling(1f-7))
    wT_bottoms = [wT_bottom₁, wT_bottom₂, wT_bottom₃, wT_bottom₄, wT_bottom₅, wT_bottom₆]


    fs = [1f-4, 1f-4, 1f-4, 0f0, 1f-4, 1f-4]

    ps = [[fs[i], uw_tops[i], uw_bottoms[i], vw_tops[i], vw_bottoms[i], wT_tops[i], wT_bottoms[i]] for i in 1:length(𝒟tests)]

    D_cell = Float32.(Dᶜ(Nz, 1/Nz))

    function NDE_nondimensional!(dx, x, p, t)
        f, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = p
        A = - τ / H
        B = f * τ
        u = x[1:Nz]
        v = x[Nz+1:2Nz]
        T = x[2Nz+1:3Nz]
        dx[1:Nz] .= A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NDE, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v)
        dx[Nz+1:2Nz] .= A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NDE, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx[2Nz+1:3Nz] .= A .* σ_wT ./ σ_T .* D_cell * predict_NDE(wT_NDE, x, wT_top, wT_bottom)
    end

    t_test = Float32.(𝒟train.t[:,1][trange]./τ)
    tspan_test = (t_test[1], t_test[end])

    uvT₀s = [𝒟test.uvT_scaled[:, trange[1]] for 𝒟test in 𝒟tests]
    uvT_tests = [𝒟test.uvT_scaled[:, trange] for 𝒟test in 𝒟tests]

    opt_NDE = ROCK4()

    probs = [ODEProblem(NDE_nondimensional!, uvT₀s[i], tspan_test, ps[i], saveat=t_test) for i in 1:length(𝒟tests)]

    function loss_NDE(prob, uvT_test)
        sol = Array(solve(prob, opt_NDE, saveat=t_test))
        loss = Flux.mse(sol, uvT_test)
        return loss
    end

    output = [loss_NDE(probs[i], uvT_tests[i]) for i in 1:length(𝒟tests)]
end

train_files = ["strong_wind"]
𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, animate=false, animate_dir="$(output_gif_directory)/Training")

output_interpolation = test_NDE(𝒟train, uw_NDE, vw_NDE, wT_NDE, 1:1:289)
# output_extrapolation = test_NDE(𝒟train, uw_NDE, vw_NDE, wT_NDE, 100:1:289)

test_datasets = ["SW, NH", "SW, WH", "SW, WC", "SW, NR", "FC", "WW, SC"]

scatter([test_datasets[1]], [output_interpolation[1]], yscale=:log10, label="Extrapolation")
scatter!(test_datasets[2:end], output_extrapolation[2:end], label="Prediction")
xlabel!("Datasets")
ylabel!("L2 Loss")
savefig("Output/loss_SWNH_comparison.pdf")