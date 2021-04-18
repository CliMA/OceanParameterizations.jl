function prepare_time_window(t, trange)
    return t[trange]
end

function prepare_training_data(uvT, trange)
    return uvT[:,trange]
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
    H = abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1])
    τ = abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1])
    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]
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

function prepare_parameters_NDE_training_unscaled(𝒟train, uw_NN, vw_NN, wT_NN, f=1f-4)
    uw_weights, re_uw = Flux.destructure(uw_NN)
    vw_weights, re_vw = Flux.destructure(vw_NN)
    wT_weights, re_wT = Flux.destructure(wT_NN)
    weights = Float32[uw_weights; vw_weights; wT_weights]
    Nz = length(𝒟train.u.z)
    Δz =  𝒟train.u.z[2] - 𝒟train.u.z[1]
    D_cell = Float32.(Dᶜ(length(𝒟train.u.z), Δz))
    D_face = Float32.(Dᶠ(length(𝒟train.u.z), Δz))
    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)
    uw_range = 1:size_uw_NN
    vw_range = size_uw_NN + 1:size_uw_NN + size_vw_NN
    wT_range = size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN
    return f, Nz, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range
end

function train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage; 
                    n_simulations, maxiters=500, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=1.67f-4, g=9.81f0, 
                    modified_pacanowski_philander=false, convective_adjustment=false, smooth_profile=false, smooth_NN=false, smooth_Ri=false, train_gradient=false,
                    zero_weights=false)
    f, H, τ, Nz, u_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN)

    @assert !modified_pacanowski_philander || !convective_adjustment

    if zero_weights
        @assert modified_pacanowski_philander
    end

    tanh_step(x) = (1 - tanh(x)) / 2

    ϵ = 1f-7

    if smooth_profile
        filter_cell = WindMixing.smoothing_filter(Nz, 3)
    end

    if smooth_NN
        if zero_weights
            filter_face = WindMixing.smoothing_filter(Nz+1, 3) 
        else
            filter_interior = WindMixing.smoothing_filter(Nz-1, 3) 
        end
    end

    if smooth_Ri
        filter_face = WindMixing.smoothing_filter(Nz+1, 3) 
    end

    function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, σ_u, σ_v, σ_T, H, g, α)
        Bz = H * g * α * σ_T * ∂T∂z
        S² = (σ_u * ∂u∂z) ^2 + (σ_v * ∂v∂z) ^2
        # if Bz == 0 && S² == 0
        #     return 0
        # else
        #     return Bz / S²
        # end
        return Bz / S²
    
    end

    function predict_NDE(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        if smooth_profile
            x[1:Nz] = filter_cell * x[1:Nz]
            x[Nz + 1:2Nz] = filter_cell * x[Nz + 1:2Nz]
            x[2Nz + 1:3Nz] = filter_cell * x[2Nz + 1:3Nz]
        end

        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:3Nz]
        
        if zero_weights
            uw = uw_NN(x)
            vw = vw_NN(x)
            wT = wT_NN(x)

            if smooth_NN
                uw = filter_face * uw
                vw = filter_face * vw
                wT = filter_face * wT
            end
        else
            # uw_interior = uw_NN(x)
            # vw_interior = vw_NN(x)
            # wT_interior = wT_NN(x)

            uw_interior = fill(uw_scaling(0f0), 31)
            vw_interior = fill(vw_scaling(0f0), 31)
            wT_interior = fill(wT_scaling(0f0), 31)

            if smooth_NN
                uw_interior = filter_interior * uw_interior
                vw_interior = filter_interior * vw_interior
                wT_interior = filter_interior * wT_interior
            end

            uw = [uw_bottom; uw_interior; uw_top]
            vw = [vw_bottom; vw_interior; vw_top]
            wT = [wT_bottom; wT_interior; wT_top]
        end

        if modified_pacanowski_philander
            ∂u∂z = D_face * u
            ∂v∂z = D_face * v
            ∂T∂z = D_face * T
            Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, σ_u, σ_v, σ_T, H, g, α)

            if smooth_Ri
                Ri = filter_face * Ri
            end

            ν = ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)

            if zero_weights

                ν∂u∂z = [[-H * σ_uw / σ_u * (uw_bottom - uw_scaling(0f0))]; ν[2:end-1] .* ∂u∂z[2:end-1]; [-H * σ_uw / σ_u * (uw_top - uw_scaling(0f0))]]
                ν∂v∂z = [[-H * σ_vw / σ_v * (vw_bottom - vw_scaling(0f0))]; ν[2:end-1] .* ∂v∂z[2:end-1]; [-H * σ_vw / σ_v * (vw_top - vw_scaling(0f0))]]
                ν∂T∂z = [[-H * σ_wT / σ_T * (wT_bottom - wT_scaling(0f0))]; ν[2:end-1] ./ Pr .* ∂T∂z[2:end-1]; [-H * σ_wT / σ_T * (wT_top - wT_scaling(0f0))]]

                ∂z_ν∂u∂z = D_cell * ν∂u∂z
                ∂z_ν∂v∂z = D_cell * ν∂v∂z
                ∂z_ν∂T∂z = D_cell * ν∂T∂z

                ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v) .+ τ / H ^ 2 .* ∂z_ν∂u∂z
                ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u) .+ τ / H ^ 2 .* ∂z_ν∂v∂z
                ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT .+ τ / H ^ 2 .* ∂z_ν∂T∂z
            else
                ∂z_ν∂u∂z = D_cell * (ν .* ∂u∂z)
                ∂z_ν∂v∂z = D_cell * (ν .* ∂v∂z)
                ∂z_ν∂T∂z = D_cell * (ν .* ∂T∂z ./ Pr)
                ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v) .+ τ / H ^ 2 .* ∂z_ν∂u∂z
                ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u) .+ τ / H ^ 2 .* ∂z_ν∂v∂z
                ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT .+ τ / H ^ 2 .* ∂z_ν∂T∂z
            end
        elseif convective_adjustment
            ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
            ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
            ∂T∂z = D_face * T
            ∂z_κ∂T∂z = D_cell * (κ .* min.(0f0, ∂T∂z))
            ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT .+ τ / H ^2 .* ∂z_κ∂T∂z
        else
            ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
            ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
            ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT
        end

        return [∂u∂t; ∂v∂t; ∂T∂t]
    end

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    function NDE(x, p, t)
        uw_weights = p[uw_range]
        vw_weights = p[vw_range]
        wT_weights = p[wT_range]
        uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = p[wT_range[end] + 1:end]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        return predict_NDE(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
    end

    uvT₀s = [Float32.(𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]]) for i in 0:n_simulations - 1]
    t_train = prepare_time_window(𝒟train.t[:,1], tsteps)
    uvT_trains = [prepare_training_data(𝒟train.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)], tsteps) for i in 0:n_simulations - 1]

    # D_face_uvT = [D_face; D_face; D_face]
    # function calculate_gradient(uvTs)
    #     Nzf = Nz + 1
    #     gradients = [zeros(Float32, size(uvT, 1) + 3, size(uvT, 2)) for uvT in uvTs]
    #     for i in 1:length(gradients)
    #         gradient = gradients[i]
    #         uvT = uvTs[i]
    #         for j in 1:size(gradient, 2)
    #             # ∂u∂z = @view gradient[1:Nzf, j]
    #             # ∂v∂z = @view gradient[Nzf+1:2Nzf, j]
    #             # ∂T∂z = @view gradient[2Nzf+1:end, j]

    #             # gradient[1:Nzf, j] = D_face * uvT[1:Nz, j]
    #             # gradient[Nzf+1:2Nzf, j] = D_face * uvT[Nz+1:2Nz, j]
    #             # gradient[2Nzf+1:end, j] = D_face * uvT[2Nz+1:3Nz, j]
    #             gradient[:,j] = D_face_uvT * uvT[:,j]
    #         end
    #     end
    #     return gradients
    # end

    function calculate_gradient(uvTs)
        return cat([cat([[D_face * uvT[1:Nz, i]; D_face * uvT[Nz+1:2Nz, i]; D_face * uvT[2Nz+1:3Nz, i]] for i in 1:size(uvT, 2)]..., dims=2) for uvT in uvTs]..., dims=2)
    end

    if train_gradient
        uvT_gradients = calculate_gradient(uvT_trains)
    end

    t_train = t_train ./ τ
    tspan_train = (t_train[1], t_train[end])
    BCs = [[𝒟train.uw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.uw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function loss(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        return mean(Flux.mse.(sols, uvT_trains))
    end

    function loss_gradient(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]
        loss_profile = mean(Flux.mse.(sols, uvT_trains))
        loss_gradient = mean(Flux.mse.(calculate_gradient(sols), uvT_gradients))
        return mean([loss_profile, loss_gradient])
    end

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient, GalacticOptim.AutoZygote())
        prob_loss = OptimizationProblem(f_loss, weights, BCs)
    else
        f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
        prob_loss = OptimizationProblem(f_loss, weights, BCs)
    end

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

function train_NDE_unscaled(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage; 
                    n_simulations, maxiters=500, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=1.67f-4, g=9.81f0, 
                    modified_pacanowski_philander=false, convective_adjustment=false, smooth_profile=false, smooth_NN=false, smooth_Ri=false, 
                    train_gradient=false, zero_weights=true)
    
    f, Nz, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters_NDE_training_unscaled(𝒟train, uw_NN, vw_NN, wT_NN)

    @assert !modified_pacanowski_philander || !convective_adjustment

    tanh_step(x) = (1 - tanh(x)) / 2

    ϵ = 1f-7

    if smooth_profile
        filter_cell = WindMixing.smoothing_filter(Nz, 3)
    end

    if smooth_NN
       filter_interior = WindMixing.smoothing_filter(Nz-1, 3) 
    end

    if smooth_Ri
        filter_face = WindMixing.smoothing_filter(Nz+1, 3) 
    end

    function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, g, α)
        Bz = g * α * ∂T∂z
        S² = ∂u∂z ^2 + ∂v∂z ^2
        return Bz / S²
    end

    function predict_NDE(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        if smooth_profile
            x[1:Nz] = filter_cell * x[1:Nz]
            x[Nz + 1:2Nz] = filter_cell * x[Nz + 1:2Nz]
            x[2Nz + 1:3Nz] = filter_cell * x[2Nz + 1:3Nz]
        end

        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:3Nz]
        
        uw_interior = uw_NN(x)
        vw_interior = vw_NN(x)
        wT_interior = wT_NN(x)

        if smooth_NN
            uw_interior = filter_interior * uw_interior
            vw_interior = filter_interior * vw_interior
            wT_interior = filter_interior * wT_interior
        end

        uw = [uw_top; uw_interior; uw_bottom]
        vw = [vw_top; vw_interior; vw_bottom]
        wT = [wT_top; wT_interior; wT_bottom]

        if modified_pacanowski_philander
            ∂u∂z = D_face * u
            ∂v∂z = D_face * v
            ∂T∂z = D_face * T
            Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, g, α)

            if smooth_Ri
                Ri = filter_face * Ri
            end

            ν = ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)
            ν∂u∂z = ν .* ∂u∂z
            # ν∂u∂z[1] = uw_top 
            ν∂v∂z = ν .* ∂v∂z
            ν∂T∂z = ν .* ∂T∂z ./ Pr
            ∂u∂t = - D_cell * (uw .- ν∂u∂z) .+ f .* v
            ∂v∂t = - D_cell * (vw .- ν∂v∂z) .- f .* u
            ∂T∂t = - D_cell * (wT .- ν∂T∂z)
        elseif convective_adjustment
            ∂u∂t = - D_cell * uw .+ f .* v
            ∂v∂t = - D_cell * vw .- f .* u

            ∂T∂z = D_face * T
            κ∂T∂z = κ .* min.(0f0, ∂T∂z)
            ∂T∂t = - D_cell * (wT .- ∂z_κ∂T∂z)
        else
            ∂u∂t = - D_cell * uw .+ f .* v
            ∂v∂t = - D_cell * vw .- f .* u
            ∂T∂t = - D_cell * wT
        end

        return [∂u∂t; ∂v∂t; ∂T∂t]
    end

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    function NDE(x, p, t)
        uw_weights = p[uw_range]
        vw_weights = p[vw_range]
        wT_weights = p[wT_range]
        uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = p[wT_range[end] + 1:end]
        uw_NN = re_uw(uw_weights)
        vw_NN = re_vw(vw_weights)
        wT_NN = re_wT(wT_weights)
        return predict_NDE(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
    end

    uvT₀s = [Float32.(𝒟train.uvT_unscaled[:,n_steps * i + tsteps[1]]) for i in 0:n_simulations - 1]
    t_train = prepare_time_window(𝒟train.t[:,1], tsteps)
    uvT_trains = [prepare_training_data(𝒟train.uvT_unscaled[:,n_steps * i + 1:n_steps * (i + 1)], tsteps) for i in 0:n_simulations - 1]

    function calculate_gradient(uvTs)
        return cat([cat([[D_face * uvT[1:Nz, i]; D_face * uvT[Nz+1:2Nz, i]; D_face * uvT[2Nz+1:3Nz, i]] for i in 1:size(uvT, 2)]..., dims=2) for uvT in uvTs]..., dims=2)
    end

    if train_gradient
        uvT_gradients = calculate_gradient(uvT_trains)
    end

    tspan_train = (t_train[1], t_train[end])
    BCs = [[𝒟train.uw.coarse[1,n_steps * i + tsteps[1]],
            𝒟train.uw.coarse[end,n_steps * i + tsteps[1]],
            𝒟train.vw.coarse[1,n_steps * i + tsteps[1]],
            𝒟train.vw.coarse[end,n_steps * i + tsteps[1]],
            𝒟train.wT.coarse[1,n_steps * i + tsteps[1]],
            𝒟train.wT.coarse[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function loss(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        return mean(Flux.mse.(sols, uvT_trains))
    end

    function loss_gradient(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]
        loss_profile = mean(Flux.mse.(sols, uvT_trains))
        loss_gradient = mean(Flux.mse.(calculate_gradient(sols), uvT_gradients))
        return mean([loss_profile, loss_gradient])
    end

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient, GalacticOptim.AutoZygote())
        prob_loss = OptimizationProblem(f_loss, weights, BCs)
    else
        f_loss = OptimizationFunction(loss, GalacticOptim.AutoZygote())
        prob_loss = OptimizationProblem(f_loss, weights, BCs)
    end

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
