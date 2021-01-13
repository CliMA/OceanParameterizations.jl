function predict_NDE(NN, x, top, bottom)
    interior = NN(x)
    return [top; interior; bottom]
end

function predict_NDE_convective_adjustment(NN, x, top, bottom, D_face, D_cell, κ, Nz)
    interior = NN(x)
    T = @view x[2Nz + 1:3Nz]
    wT = [top; interior; bottom]
    ∂T∂z = D_face * T
    ∂z_κ∂T∂z = D_cell * min.(0f0, κ .* ∂T∂z)
    return - D_cell * wT .+ ∂z_κ∂T∂z
end

function prepare_time_window(t, trange)
    return Float32.(t[trange])
end

function prepare_training_data(uvT, trange)
    return Float32.(uvT[:,trange])
end

function save_NDE_weights(weights, size_uw_NN, size_vw_NN, size_wT_NN, FILE_PATH=pwd(), filename="weights")
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

function prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f=1f-4, Nz=32)
    H = Float32(abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1]))
    τ = Float32(abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1]))
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
    weights = Float32[uw_weights; vw_weights; wT_weights]
    D_cell = Float32.(Dᶜ(Nz, 1 / Nz))
    D_face = Float32.(Dᶠ(Nz, 1 / Nz))
    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)
    uw_range = 1:size_uw_NN
    vw_range = size_uw_NN + 1:size_uw_NN + size_vw_NN
    wT_range = size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN
    return f, H, τ, Nz, u_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range
end

function train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage, n_simulations, maxiters=500)
    f, H, τ, Nz, u_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN)

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    κ = 10f0

    function NDE!(dx, x, p, t)
        uw_weights = p[uw_range]
        vw_weights = p[vw_range]
        wT_weights = p[wT_range]
        uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = p[wT_range[end] + 1:end]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        A = - τ / H
        B = f * τ
        u = x[1:Nz]
        v = x[Nz + 1:2Nz]
        T = x[2Nz + 1:3Nz]
        dx[1:Nz] .= A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NN, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) # nondimensional gradient
        dx[Nz + 1:2Nz] .= A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NN, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx[2Nz + 1:3Nz] .= A .* σ_wT ./ σ_T .* predict_NDE(wT_NN, x, wT_top, wT_bottom)
    end

    uvT₀s = [Float32.(𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]]) for i in 0:n_simulations - 1]
    t_train = prepare_time_window(𝒟train.t[:,1], tsteps)
    uvT_trains = [prepare_training_data(𝒟train.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)], tsteps) for i in 0:n_simulations - 1]
    t_train = Float32.(t_train ./ τ)
    tspan_train = (t_train[1], t_train[end])
    BCs = [[Float32.(𝒟train.uw.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.uw.scaled[end,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.vw.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.vw.scaled[end,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.wT.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.wT.scaled[end,n_steps * i + tsteps[1]])] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(NDE!, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function loss(weights, BCs)
        sols = [Float32.(Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(), saveat=t_train))) for i in 1:n_simulations]
        return mean(Flux.mse.(sols, uvT_trains))
    end

    f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob_loss = OptimizationProblem(f_loss, weights, BCs)

    for i in 1:length(optimizers), epoch in 1:epochs
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                @info "NDE, loss = $(args[2]), stage $stage, optimizer $i/$(length(optimizers)), epoch $epoch/$epochs, iteration = $iter/$maxiters"
                write_data_NDE_training(FILE_PATH, args[2], re_uw(args[1][uw_range]), re_vw(args[1][vw_range]), re_wT(args[1][wT_range]), stage)
            end
            iter += 1
            false
        end
        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        weights .= res.minimizer
    end
    return re_uw(weights[uw_range]), re_vw(weights[vw_range]), re_wT(weights[wT_range])
end

function train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage, n_simulations, maxiters=500)
    f, H, τ, Nz, u_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN)

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    κ = 10f0

    function NDE!(dx, x, p, t)
        uw_weights = p[uw_range]
        vw_weights = p[vw_range]
        wT_weights = p[wT_range]
        uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = p[wT_range[end] + 1:end]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        A = - τ / H
        B = f * τ
        u = x[1:Nz]
        v = x[Nz + 1:2Nz]
        T = x[2Nz + 1:3Nz]
        dx[1:Nz] .= A .* σ_uw ./ σ_u .* D_cell * predict_NDE(uw_NN, x, uw_top, uw_bottom) .+ B ./ σ_u .* (σ_v .* v .+ μ_v) # nondimensional gradient
        dx[Nz + 1:2Nz] .= A .* σ_vw ./ σ_v .* D_cell * predict_NDE(vw_NN, x, vw_top, vw_bottom) .- B ./ σ_v .* (σ_u .* u .+ μ_u)
        dx[2Nz + 1:3Nz] .= -A .* σ_wT ./ σ_T .* predict_NDE_convective_adjustment(wT_NN, x, wT_top, wT_bottom, D_face, D_cell, κ, Nz)
    end

    uvT₀s = [Float32.(𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]]) for i in 0:n_simulations - 1]
    t_train = prepare_time_window(𝒟train.t[:,1], tsteps)
    uvT_trains = [prepare_training_data(𝒟train.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)], tsteps) for i in 0:n_simulations - 1]
    t_train = Float32.(t_train ./ τ)
    tspan_train = (t_train[1], t_train[end])
    BCs = [[Float32.(𝒟train.uw.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.uw.scaled[end,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.vw.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.vw.scaled[end,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.wT.scaled[1,n_steps * i + tsteps[1]]),
            Float32.(𝒟train.wT.scaled[end,n_steps * i + tsteps[1]])] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(NDE!, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function loss(weights, BCs)
        sols = [Float32.(Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(), saveat=t_train))) for i in 1:n_simulations]
        return mean(Flux.mse.(sols, uvT_trains))
    end

    f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob_loss = OptimizationProblem(f_loss, weights, BCs)

    for i in 1:length(optimizers), epoch in 1:epochs
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                @info "loss = $(args[2]), stage $stage, optimizer $i/$(length(optimizers)), epoch $epoch/$epochs, iteration = $iter/$maxiters"
                write_data_NDE_training(FILE_PATH, args[2], re_uw(args[1][uw_range]), re_vw(args[1][vw_range]), re_wT(args[1][wT_range]), stage)
            end
            iter += 1
            false
        end
        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        weights .= res.minimizer
    end
    return re_uw(weights[uw_range]), re_vw(weights[vw_range]), re_wT(weights[wT_range])
end