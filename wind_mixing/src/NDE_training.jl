using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using GalacticOptim

include("data_containers.jl")

function predict_NDE(NN, x, top, bottom)
    interior = NN(x)
    return [top; interior; bottom]
end

function time_window(t, uvT, trange)
    return (Float32.(t[trange]), Float32.(uvT[:,trange]))
end

function save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, FILE_PATH, filename)
    uw_weights = weights[1:size_uw_NN]
    vw_weights = weights[size_uw_NN + 1:size_uw_NN + size_vw_NN]
    wT_weights = weights[size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN]
    uw_NN_params = Dict(:weights => uw_weights)
    bson(joinpath(FILE_PATH, "uw_$filename.bson"), uw_NN_params)

    vw_NN_params = Dict(:weights => vw_weights)
    bson(joinpath(FILE_PATH, "vw_$filename.bson"), vw_NN_params)

    wT_NN_params = Dict(:weights => wT_weights)
    bson(joinpath(FILE_PATH, "wT_$filename.bson"), wT_NN_params)
end

function cb(args...)
    @info "loss = $(args[2])"
    false
end

function train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, OUTPUT_PATH, filename)
    f = 1f-4
    H = Float32(abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1]))
    τ = Float32(abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1]))
    Nz = 32
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
    uw_weights, re_uw = Flux.destructure(uw_NN)
    vw_weights, re_vw = Flux.destructure(vw_NN)
    wT_weights, re_wT = Flux.destructure(wT_NN)
    uw_top = Float32(𝒟train.uw.scaled[1,1])
    uw_bottom = Float32(𝒟train.uw.scaled[end,1])
    vw_top = Float32(𝒟train.vw.scaled[1,1])
    vw_bottom = Float32(𝒟train.vw.scaled[end,1])
    wT_top = Float32(𝒟train.wT.scaled[1,1])
    wT_bottom = Float32(𝒟train.wT.scaled[end,1])

    D_cell = Float32.(Dᶜ(Nz, 1/Nz))

    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)
    weights = Float32[uw_weights; vw_weights; wT_weights]

    function NDE!(dx, x, p, t)
        uw_weights = p[1:size_uw_NN]
        vw_weights = p[size_uw_NN + 1:size_uw_NN + size_vw_NN]
        wT_weights = p[size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        A = - τ / H
        B = f * τ
        u = x[1:Nz]
        v = x[Nz+1:2*Nz]
        T = x[2*Nz+1:96]
        dx[1:Nz] .= A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NN, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
        dx[Nz+1:2Nz] .= A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NN, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx[2Nz+1:3Nz] .= A .* σ_wT ./ σ_T .* D_cell * predict_NDE(wT_NN, x, wT_top, wT_bottom)
    end

    uvT₀ = Float32.(𝒟train.uvT_scaled[:,tsteps[1]])
    t_train, uvT_train = time_window(𝒟train.t, 𝒟train.uvT_scaled, tsteps)
    t_train = Float32.(t_train ./ τ)
    tspan_train = (t_train[1], t_train[end])

    prob_NDE = ODEProblem(NDE!, uvT₀, tspan_train, weights, saveat=t_train)

    function loss(weights, p)
        sol = Float32.(Array(solve(prob_NDE, timestepper, p=weights, sensealg=InterpolatingAdjoint())))
        return Flux.mse(sol, uvT_train)
    end

    f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob_loss = OptimizationProblem(f_loss, weights)

    for opt in optimizers, epoch in 1:epochs
        @info "Epoch $epoch, $opt"
        res = solve(prob_loss, opt, cb=cb, maxiters = 500)
        weights .= res.minimizer
        save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, OUTPUT_PATH, filename)
    end
    save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, OUTPUT_PATH, filename)
end

function train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, OUTPUT_PATH, filename)
    f = 1f-4
    H = Float32(abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1]))
    τ = Float32(abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1]))
    Nz = 32
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
    uw_weights, re_uw = Flux.destructure(uw_NN)
    vw_weights, re_vw = Flux.destructure(vw_NN)
    wT_weights, re_wT = Flux.destructure(wT_NN)
    uw_top = Float32(𝒟train.uw.scaled[1,1])
    uw_bottom = Float32(𝒟train.uw.scaled[end,1])
    vw_top = Float32(𝒟train.vw.scaled[1,1])
    vw_bottom = Float32(𝒟train.vw.scaled[end,1])
    wT_top = Float32(𝒟train.wT.scaled[1,1])
    wT_bottom = Float32(𝒟train.wT.scaled[end,1])

    κ = 10f0

    D_cell = Float32.(Dᶜ(Nz, 1/Nz))
    D_face = Float32.(Dᶠ(Nz, 1/Nz))

    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)
    weights = Float32[uw_weights; vw_weights; wT_weights]

    function predict_NDE(NN, x, top, bottom)
        interior = NN(x)
        return [top; interior; bottom]
    end

    function predict_NDE_convective_adjustment(NN, x, top, bottom)
        interior = NN(x)
        T = @view x[2Nz + 1:3Nz]
        wT = [top; interior; bottom]
        ∂T∂z = D_face * T
        ∂z_κ∂T∂z = D_cell * min.(0f0, κ .* ∂T∂z)
        return - D_cell * wT .+ ∂z_κ∂T∂z
    end    

    function NDE!(dx, x, p, t)
        uw_weights = p[1:size_uw_NN]
        vw_weights = p[size_uw_NN + 1:size_uw_NN + size_vw_NN]
        wT_weights = p[size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        A = - τ / H
        B = f * τ
        u = x[1:Nz]
        v = x[Nz+1:2*Nz]
        T = x[2*Nz+1:96]
        dx[1:Nz] .= A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NN, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) #nondimensional gradient
        dx[Nz+1:2Nz] .= A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NN, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx[2Nz+1:3Nz] .= -A .* σ_wT ./ σ_T .* predict_NDE_convective_adjustment(wT_NN, x, wT_top, wT_bottom)
    end

    uvT₀ = Float32.(𝒟train.uvT_scaled[:,tsteps[1]])
    t_train, uvT_train = time_window(𝒟train.t, 𝒟train.uvT_scaled, tsteps)
    t_train = Float32.(t_train ./ τ)
    tspan_train = (t_train[1], t_train[end])

    prob_NDE = ODEProblem(NDE!, uvT₀, tspan_train, weights, saveat=t_train)

    function loss(weights, p)
        sol = Array(solve(prob_NDE, timestepper, p=weights, sensealg=InterpolatingAdjoint()))
        return Flux.mse(sol, uvT_train)
    end

    f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob_loss = OptimizationProblem(f_loss, weights)

    for opt in optimizers, epoch in 1:epochs
        @info "Epoch $epoch, $opt"
        res = solve(prob_loss, opt, cb=cb, maxiters=500)
        weights .= res.minimizer
        save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, OUTPUT_PATH, filename)
    end
    save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, OUTPUT_PATH, filename)
end

