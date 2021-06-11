

function prepare_parameters_NDE_animation(𝒟train, uw_NN, vw_NN, wT_NN, f=1f-4, Nz=32)
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
    return f, H, τ, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range
end

function prepare_BCs(𝒟, scalings)
    uw_top = scalings.uw(𝒟.uw.coarse[end,1])
    uw_bottom = scalings.uw(𝒟.uw.coarse[1,1])
    vw_top = scalings.vw(𝒟.vw.coarse[end,1])
    vw_bottom = scalings.vw(𝒟.vw.coarse[1,1])
    wT_top = scalings.wT(𝒟.wT.coarse[end,1])
    wT_bottom = scalings.wT(𝒟.wT.coarse[1,1])
    return (uw=(top=uw_top, bottom=uw_bottom), vw=(top=vw_top, bottom=vw_bottom), wT=(top=wT_top, bottom=wT_bottom))
end

function NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange; 
                    unscale=true, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=1.67f-4, g=9.81f0, f=1f-4,
                    modified_pacanowski_philander=false, convective_adjustment=false,
                    smooth_NN=false, smooth_Ri=false,
                    zero_weights=false, 
                    gradient_scaling = 5f-3)
    
    @assert !modified_pacanowski_philander || !convective_adjustment

    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                    smooth_NN=smooth_NN, smooth_Ri=smooth_Ri,
                    zero_weights=zero_weights)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)

    H, τ, f = constants.H, constants.τ, constants.f
    D_face, D_cell = derivatives.face, derivatives.cell

    BCs = prepare_BCs(𝒟test, scalings)
    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = BCs.uw.bottom, BCs.uw.top, BCs.vw.bottom, BCs.vw.top, BCs.wT.bottom, BCs.wT.top

    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)


    if modified_pacanowski_philander
        constants_NN_only = (H=constants.H, τ=constants.τ, f=constants.f, Nz=constants.Nz, g=constants.g, α=constants.α, ν₀=0f0, ν₋=0f0, Riᶜ=constants.Riᶜ, ΔRi=constants.ΔRi, Pr=constants.Pr)
    end


    t_test = Float32.(𝒟test.t[trange] ./ constants.τ)
    tspan_test = (t_test[1], t_test[end])
    uvT₀ = [scalings.u(𝒟test.uvT_unscaled[1:Nz, 1]); scalings.v(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); scalings.T(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]
    prob = ODEProblem(prob_NDE, uvT₀, tspan_test)
    sol = Array(solve(prob, ROCK4(), p=[weights; uw_bottom; uw_top; vw_bottom; vw_top; wT_bottom; wT_top], saveat=t_test))

    if modified_pacanowski_philander
        sol_modified_pacanowski_philander = Array(solve(prob, ROCK4(), p=[zeros(Float32, length(weights)); uw_bottom; uw_top; vw_bottom; vw_top; wT_bottom; wT_top], saveat=t_test))
    end

    output = Dict()

    𝒟test_uvT_scaled = [scalings.u.(𝒟test.uvT_unscaled[1:Nz, trange]); 
                        scalings.v.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]); 
                        scalings.T.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])]

    𝒟test_uvT_scaled_gradient = calculate_profile_gradient(𝒟test_uvT_scaled, derivatives, constants)

    losses = [loss(@view(sol[:,i]), @view(𝒟test_uvT_scaled[:,i])) for i in 1:size(sol, 2)]

    sol_gradient = calculate_profile_gradient(sol, derivatives, constants)
    losses_gradient = [loss_gradient(@view(𝒟test_uvT_scaled[:,i]), 
                                     @view(sol[:,i]), 
                                     @view(𝒟test_uvT_scaled_gradient[:,i]), 
                                     @view(sol_gradient[:,i]), 
                                     gradient_scaling) for i in 1:size(sol, 2)]

    if modified_pacanowski_philander
        output["train_parameters"] = (ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, gradient_scaling=gradient_scaling)
    end

    output["losses"] = losses
    output["loss"] = mean(losses)
    output["losses_gradient"] = losses_gradient .- losses
    output["loss_gradient"] = mean(losses_gradient)

    if modified_pacanowski_philander
        sol_modified_pacanowski_philander_gradient = calculate_profile_gradient(sol_modified_pacanowski_philander, derivatives, constants)
        losses_modified_pacanowski_philander = [loss(@view(sol_modified_pacanowski_philander[:,i]), 
                                                     @view(𝒟test_uvT_scaled[:,i])) 
                                                     for i in 1:size(sol_modified_pacanowski_philander, 2)]
        losses_modified_pacanowski_philander_gradient = [loss_gradient(@view(𝒟test_uvT_scaled[:,i]), 
                                                                       @view(sol_modified_pacanowski_philander[:,i]), 
                                                                       @view(𝒟test_uvT_scaled_gradient[:,i]), 
                                                                       @view(sol_modified_pacanowski_philander_gradient[:,i]), 
                                                                       gradient_scaling) for i in 1:size(sol_modified_pacanowski_philander, 2)]
        output["losses_modified_pacanowski_philander"] = losses_modified_pacanowski_philander
        output["loss_modified_pacanowski_philander"] = mean(losses_modified_pacanowski_philander)
        output["losses_modified_pacanowski_philander_gradient"] = losses_modified_pacanowski_philander_gradient .- losses_modified_pacanowski_philander
        output["loss_modified_pacanowski_philander_gradient"] = mean(losses_modified_pacanowski_philander_gradient)
    end

    truth_uw = 𝒟test.uw.coarse[:,trange]
    truth_vw = 𝒟test.vw.coarse[:,trange]
    truth_wT = 𝒟test.wT.coarse[:,trange]
    
    truth_u = 𝒟test.uvT_unscaled[1:Nz, trange]
    truth_v = 𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]
    truth_T = 𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange]

    test_uw = similar(truth_uw)
    test_vw = similar(truth_vw)
    test_wT = similar(truth_wT)

    for i in 1:size(test_uw, 2)
        test_uw[:,i], test_vw[:,i], test_wT[:,i] = predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
    end

    test_uw .= inv(scalings.uw).(test_uw)
    test_vw .= inv(scalings.vw).(test_vw)
    test_wT .= inv(scalings.wT).(test_wT)
    test_u = inv(scalings.u).(sol[1:Nz,:])
    test_v = inv(scalings.v).(sol[Nz + 1:2Nz, :])
    test_T = inv(scalings.T).(sol[2Nz + 1: 3Nz, :])

    depth_profile = 𝒟test.u.z
    depth_flux = 𝒟test.uw.z
    t = 𝒟test.t[trange]

    truth_Ri = similar(𝒟test.uw.coarse[:,trange])

    for i in 1:size(truth_Ri, 2)
        truth_Ri[:,i] .= local_richardson.(D_face * 𝒟test.u.scaled[:,i], D_face * 𝒟test.v.scaled[:,i], D_face * 𝒟test.T.scaled[:,i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end

    test_Ri = similar(truth_Ri)

    for i in 1:size(test_Ri,2)
        test_Ri[:,i] .= local_richardson.(D_face * sol[1:Nz,i], D_face * sol[Nz + 1:2Nz, i], D_face * sol[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end

    output["truth_Ri"] = truth_Ri
    output["test_Ri"] = test_Ri

    if modified_pacanowski_philander
        test_uw_modified_pacanowski_philander = similar(truth_uw)
        test_vw_modified_pacanowski_philander = similar(truth_vw)
        test_wT_modified_pacanowski_philander = similar(truth_wT)

        for i in 1:size(test_uw_modified_pacanowski_philander, 2)
            test_uw_modified_pacanowski_philander[:,i], test_vw_modified_pacanowski_philander[:,i], test_wT_modified_pacanowski_philander[:,i] = 
                                    predict_flux(NN_constructions.uw(zeros(Float32, NN_sizes.uw)), 
                                                NN_constructions.vw(zeros(Float32, NN_sizes.vw)), 
                                                NN_constructions.wT(zeros(Float32, NN_sizes.wT)), 
                                     @view(sol_modified_pacanowski_philander[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
        end

        test_uw_modified_pacanowski_philander .= inv(scalings.uw).(test_uw_modified_pacanowski_philander)
        test_vw_modified_pacanowski_philander .= inv(scalings.vw).(test_vw_modified_pacanowski_philander)
        test_wT_modified_pacanowski_philander .= inv(scalings.wT).(test_wT_modified_pacanowski_philander)
        test_u_modified_pacanowski_philander = inv(scalings.u).(sol_modified_pacanowski_philander[1:Nz,:])
        test_v_modified_pacanowski_philander = inv(scalings.v).(sol_modified_pacanowski_philander[Nz + 1:2Nz, :])
        test_T_modified_pacanowski_philander = inv(scalings.T).(sol_modified_pacanowski_philander[2Nz + 1: 3Nz, :])

        test_Ri_modified_pacanowski_philander = similar(truth_Ri)

        for i in 1:size(test_Ri_modified_pacanowski_philander,2)
            test_Ri_modified_pacanowski_philander[:,i] .= 
            local_richardson.(D_face * sol_modified_pacanowski_philander[1:Nz,i], 
                            D_face * sol_modified_pacanowski_philander[Nz + 1:2Nz, i], 
                            D_face * sol_modified_pacanowski_philander[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
        end

        test_uw_NN_only = similar(truth_uw)
        test_vw_NN_only = similar(truth_vw)
        test_wT_NN_only = similar(truth_wT)

        for i in 1:size(test_uw_NN_only, 2)
            test_uw_NN_only[:,i], test_vw_NN_only[:,i], test_wT_NN_only[:,i] = 
            predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), BCs, conditions, scalings, constants_NN_only, derivatives, filters)
        end

        test_uw_NN_only .= inv(scalings.uw).(test_uw_NN_only)
        test_vw_NN_only .= inv(scalings.vw).(test_vw_NN_only)
        test_wT_NN_only .= inv(scalings.wT).(test_wT_NN_only)

        output["test_Ri_modified_pacanowski_philander"] = test_Ri_modified_pacanowski_philander
    end

    if !unscale
        truth_uw .= scalings.uw.(𝒟test.uw.coarse[:,trange])
        truth_vw .= scalings.vw.(𝒟test.vw.coarse[:,trange])
        truth_wT .= scalings.wT.(𝒟test.wT.coarse[:,trange])

        truth_u .= scalings.u.(𝒟test.uvT_unscaled[1:Nz, trange])
        truth_v .= scalings.v.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange])
        truth_T .= scalings.T.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])

        test_uw .= scalings.uw.(test_uw)
        test_vw .= scalings.vw.(test_vw)
        test_wT .= scalings.wT.(test_wT)

        test_u .= scalings.u.(test_u)
        test_v .= scalings.v.(test_v)
        test_T .= scalings.w.(test_T)

        if modified_pacanowski_philander
            test_uw_modified_pacanowski_philander .= scalings.uw.(test_uw_modified_pacanowski_philander)
            test_vw_modified_pacanowski_philander .= scalings.vw.(test_vw_modified_pacanowski_philander)
            test_wT_modified_pacanowski_philander .= scalings.wT.(test_wT_modified_pacanowski_philander)
    
            test_u_modified_pacanowski_philander .= scalings.u.(test_u_modified_pacanowski_philander)
            test_v_modified_pacanowski_philander .= scalings.v.(test_v_modified_pacanowski_philander)
            test_T_modified_pacanowski_philander .= scalings.w.(test_T_modified_pacanowski_philander)

            test_uw_NN_only .= scalings.uw.(test_uw_NN_only)
            test_vw_NN_only .= scalings.vw.(test_vw_NN_only)
            test_wT_NN_only .= scalings.wT.(test_wT_NN_only)
        end
    end

    if unscale
        test_uw .= test_uw .- test_uw[1, 1]
        test_vw .= test_vw .- test_vw[1, 1] 
        test_wT .= test_wT .- test_wT[1, 1]

        if modified_pacanowski_philander
            test_uw_modified_pacanowski_philander .= test_uw_modified_pacanowski_philander .- test_uw_modified_pacanowski_philander[1, 1]
            test_vw_modified_pacanowski_philander .= test_vw_modified_pacanowski_philander .- test_vw_modified_pacanowski_philander[1, 1] 
            test_wT_modified_pacanowski_philander .= test_wT_modified_pacanowski_philander .- test_wT_modified_pacanowski_philander[1, 1]

            test_uw_NN_only .= test_uw_NN_only .- test_uw_NN_only[1, 1]
            test_vw_NN_only .= test_vw_NN_only .- test_vw_NN_only[1, 1] 
            test_wT_NN_only .= test_wT_NN_only .- test_wT_NN_only[1, 1]
        end
    end

    output["truth_uw"] = truth_uw
    output["truth_vw"] = truth_vw
    output["truth_wT"] = truth_wT

    output["truth_u"] = truth_u
    output["truth_v"] = truth_v
    output["truth_T"] = truth_T

    output["test_uw"] = test_uw
    output["test_vw"] = test_vw
    output["test_wT"] = test_wT

    output["test_u"] = test_u
    output["test_v"] = test_v
    output["test_T"] = test_T

    output["depth_profile"] = 𝒟test.u.z
    output["depth_flux"] = 𝒟test.uw.z
    output["t"] = 𝒟test.t[trange]

    if modified_pacanowski_philander
        output["test_uw_modified_pacanowski_philander"] = test_uw_modified_pacanowski_philander
        output["test_vw_modified_pacanowski_philander"] = test_vw_modified_pacanowski_philander
        output["test_wT_modified_pacanowski_philander"] = test_wT_modified_pacanowski_philander
    
        output["test_u_modified_pacanowski_philander"] = test_u_modified_pacanowski_philander
        output["test_v_modified_pacanowski_philander"] = test_v_modified_pacanowski_philander
        output["test_T_modified_pacanowski_philander"] = test_T_modified_pacanowski_philander

        output["test_uw_NN_only"] = test_uw_NN_only
        output["test_vw_NN_only"] = test_vw_NN_only
        output["test_wT_NN_only"] = test_wT_NN_only
    end
    return output
end

function solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, timestepper)
    μ_u = scalings.u.μ
    μ_v = scalings.v.μ
    σ_u = scalings.u.σ
    σ_v = scalings.v.σ
    σ_T = scalings.T.σ
    σ_uw = scalings.uw.σ
    σ_vw = scalings.vw.σ
    σ_wT = scalings.wT.σ
    H, τ, f, Nz, g, α = constants.H, constants.τ, constants.f, constants.Nz, constants.g, constants.α
    ν₀, ν₋, Riᶜ, ΔRi, Pr = constants.ν₀, constants.ν₋, constants.Riᶜ, constants.ΔRi, constants.Pr
    D_face = derivatives.face
    D_cell = derivatives.cell

    uw = zeros(Float32, Nz+1)
    vw = similar(uw)
    wT = similar(uw)

    ∂u∂z = similar(uw)
    ∂v∂z = similar(uw)
    ∂T∂z = similar(uw)

    ν = similar(uw)

    Ri = similar(uw)

    uw[1] = BCs.uw.bottom - scalings.uw(0f0)
    vw[1] = BCs.vw.bottom - scalings.vw(0f0)
    wT[1] = BCs.wT.bottom - scalings.wT(0f0)

    uw[end] = BCs.uw.top - scalings.uw(0f0)
    vw[end] = BCs.vw.top - scalings.vw(0f0)
    wT[end] = BCs.wT.top - scalings.wT(0f0)

    uw_interior = @view uw[2:end-1]
    vw_interior = @view vw[2:end-1]
    wT_interior = @view wT[2:end-1]

    ∂uw∂z = zeros(Float32, Nz)
    ∂vw∂z = similar(∂uw∂z)
    ∂wT∂z = similar(∂uw∂z)

    dx = zeros(Float32, 3Nz)

    function predict_flux!(uvT, u, v, T)  
        uw_interior .= uw_NN(uvT)
        vw_interior .= vw_NN(uvT)
        wT_interior .= wT_NN(uvT)

        mul!(∂u∂z, D_face, u)
        mul!(∂v∂z, D_face, v)
        mul!(∂T∂z, D_face, T)

        Ri .= local_richardson.(∂u∂z, ∂v∂z, ∂T∂z, H, g, α, σ_u, σ_v, σ_T)
        ν .= ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)

        uw_interior .-= σ_u ./ σ_uw ./ H .* @view(ν[2:end-1]) .* @view(∂u∂z[2:end-1])
        vw_interior .-= σ_v ./ σ_vw ./ H .* @view(ν[2:end-1]) .* @view(∂v∂z[2:end-1])
        wT_interior .-= σ_T ./ σ_wT ./ H .* @view(ν[2:end-1]) .* @view(∂T∂z[2:end-1]) ./ Pr
    end

    function NDE!(dx, x, p, t)
        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:end]

        ∂u∂t = @view dx[1:Nz]
        ∂v∂t = @view dx[Nz+1:2Nz]
        ∂T∂t = @view dx[2Nz+1:end]

        predict_flux!(x, u, v, T)

        mul!(∂uw∂z, D_cell, uw)
        mul!(∂vw∂z, D_cell, vw)
        mul!(∂wT∂z, D_cell, wT)

        ∂u∂t .= -τ ./ H .* σ_uw ./ σ_u .* ∂uw∂z .+ f .* τ ./ σ_u .* (σ_v .* v .+ μ_v)
        ∂v∂t .= -τ ./ H .* σ_vw ./ σ_v .* ∂vw∂z .- f .* τ ./ σ_v .* (σ_u .* u .+ μ_u)
        ∂T∂t .= -τ ./ H .* σ_wT ./ σ_T .* ∂wT∂z
    end

    tspan = (ts[1], ts[end])
    prob = ODEProblem(NDE!, uvT₀, tspan)
    sol = Array(solve(prob, timestepper, saveat=ts))
    return sol
end

function solve_NDE_mutating_GPU(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, ts, tspan, timestepper)
    μ_u = scalings.u.μ
    μ_v = scalings.v.μ
    σ_u = scalings.u.σ
    σ_v = scalings.v.σ
    σ_T = scalings.T.σ
    σ_uw = scalings.uw.σ
    σ_vw = scalings.vw.σ
    σ_wT = scalings.wT.σ
    H, τ, f, Nz, g, α = constants.H, constants.τ, constants.f, constants.Nz, constants.g, constants.α
    ν₀, ν₋, Riᶜ, ΔRi, Pr = constants.ν₀, constants.ν₋, constants.Riᶜ, constants.ΔRi, constants.Pr
    D_face = derivatives.face |> gpu
    D_cell = derivatives.cell |> gpu

    uw = zeros(Float32, Nz+1)
    vw = similar(uw)
    wT = similar(uw)

    ∂u∂z = similar(uw) |> gpu
    ∂v∂z = similar(∂u∂z)
    ∂T∂z = similar(∂u∂z)

    ν = similar(∂u∂z)
    Ri = similar(∂u∂z)

    uw[1] = BCs.uw.bottom - scalings.uw(0f0)
    vw[1] = BCs.vw.bottom - scalings.vw(0f0)
    wT[1] = BCs.wT.bottom - scalings.wT(0f0)

    uw[end] = BCs.uw.top - scalings.uw(0f0)
    vw[end] = BCs.vw.top - scalings.vw(0f0)
    wT[end] = BCs.wT.top - scalings.wT(0f0)

    uw = uw |> gpu
    vw = vw |> gpu
    wT = wT |> gpu

    uw_interior = @view uw[2:end-1]
    vw_interior = @view vw[2:end-1]
    wT_interior = @view wT[2:end-1]

    ∂uw∂z = zeros(Float32, Nz) |> gpu
    ∂vw∂z = similar(∂uw∂z)
    ∂wT∂z = similar(∂uw∂z)

    # dx = zeros(Float32, 3Nz) |> gpu

    function predict_flux!(uvT, u, v, T)  
        uw_interior .= uw_NN(uvT)
        vw_interior .= vw_NN(uvT)
        wT_interior .= wT_NN(uvT)

        mul!(∂u∂z, D_face, u)
        mul!(∂v∂z, D_face, v)
        mul!(∂T∂z, D_face, T)

        Ri .= local_richardson.(∂u∂z, ∂v∂z, ∂T∂z, H, g, α, σ_u, σ_v, σ_T)
        ν .= ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)

        uw_interior .-= σ_u ./ σ_uw ./ H .* @view(ν[2:end-1]) .* @view(∂u∂z[2:end-1])
        vw_interior .-= σ_v ./ σ_vw ./ H .* @view(ν[2:end-1]) .* @view(∂v∂z[2:end-1])
        wT_interior .-= σ_T ./ σ_wT ./ H .* @view(ν[2:end-1]) .* @view(∂T∂z[2:end-1]) ./ Pr
    end

    function NDE!(dx, x, p, t)
        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:end]

        ∂u∂t = @view dx[1:Nz]
        ∂v∂t = @view dx[Nz+1:2Nz]
        ∂T∂t = @view dx[2Nz+1:end]

        predict_flux!(x, u, v, T)

        mul!(∂uw∂z, D_cell, uw)
        mul!(∂vw∂z, D_cell, vw)
        mul!(∂wT∂z, D_cell, wT)

        ∂u∂t .= -τ ./ H .* σ_uw ./ σ_u .* ∂uw∂z .+ f .* τ ./ σ_u .* (σ_v .* v .+ μ_v)
        ∂v∂t .= -τ ./ H .* σ_vw ./ σ_v .* ∂vw∂z .- f .* τ ./ σ_v .* (σ_u .* u .+ μ_u)
        ∂T∂t .= -τ ./ H .* σ_wT ./ σ_T .* ∂wT∂z
    end

    prob = ODEProblem(NDE!, uvT₀, tspan)
    sol = Array(solve(prob, timestepper, saveat=ts))
    return sol
end

function NDE_profile_mutating(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange;
                              unscale=true, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=1.67f-4, g=9.80665f0, f=1f-4,
                              OUTPUT_PATH = "",
                              modified_pacanowski_philander=false, convective_adjustment=false,
                              smooth_NN=false, smooth_Ri=false,
                              zero_weights=false, 
                              gradient_scaling = 5f-3,
                              timestepper=ROCK4())
    
    @assert !modified_pacanowski_philander || !convective_adjustment

    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                    smooth_NN=smooth_NN, smooth_Ri=smooth_Ri,
                    zero_weights=zero_weights)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)

    H, τ, f = constants.H, constants.τ, constants.f
    D_face, D_cell = derivatives.face, derivatives.cell

    BCs = prepare_BCs(𝒟test, scalings)
    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = BCs.uw.bottom, BCs.uw.top, BCs.vw.bottom, BCs.vw.top, BCs.wT.bottom, BCs.wT.top

    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)

    if modified_pacanowski_philander
        constants_NN_only = (H=constants.H, τ=constants.τ, f=constants.f, Nz=constants.Nz, g=constants.g, α=constants.α, ν₀=0f0, ν₋=0f0, Riᶜ=constants.Riᶜ, ΔRi=constants.ΔRi, Pr=constants.Pr)
    end

    t_test = Float32.(𝒟test.t[trange] ./ constants.τ)
    uvT₀ = [scalings.u(𝒟test.uvT_unscaled[1:Nz, 1]); scalings.v(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); scalings.T(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]

    sol = solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, t_test, timestepper)

    if modified_pacanowski_philander
        zeros_uw_NN = NN_constructions.uw(zeros(Float32, NN_sizes.uw))
        zeros_vw_NN = NN_constructions.vw(zeros(Float32, NN_sizes.vw))
        zeros_wT_NN = NN_constructions.wT(zeros(Float32, NN_sizes.wT))

        sol_modified_pacanowski_philander = solve_NDE_mutating(zeros_uw_NN, zeros_vw_NN, zeros_wT_NN, scalings, constants, BCs, derivatives, uvT₀, t_test, timestepper)
    end

    output = Dict()

    𝒟test_uvT_scaled = [scalings.u.(𝒟test.uvT_unscaled[1:Nz, trange]); 
                        scalings.v.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]); 
                        scalings.T.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])]

    𝒟test_uvT_scaled_gradient = calculate_profile_gradient(𝒟test_uvT_scaled, derivatives, constants)

    losses = [loss(@view(sol[:,i]), @view(𝒟test_uvT_scaled[:,i])) for i in 1:size(sol, 2)]

    sol_gradient = calculate_profile_gradient(sol, derivatives, constants)
    losses_gradient = [loss_gradient(@view(𝒟test_uvT_scaled[:,i]), 
                                     @view(sol[:,i]), 
                                     @view(𝒟test_uvT_scaled_gradient[:,i]), 
                                     @view(sol_gradient[:,i]), 
                                     gradient_scaling) for i in 1:size(sol, 2)]

    if modified_pacanowski_philander
        output["train_parameters"] = (ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, gradient_scaling=gradient_scaling)
    end

    output["losses"] = losses
    output["loss"] = mean(losses)
    output["losses_gradient"] = losses_gradient .- losses
    output["loss_gradient"] = mean(losses_gradient)

    if modified_pacanowski_philander
        sol_modified_pacanowski_philander_gradient = calculate_profile_gradient(sol_modified_pacanowski_philander, derivatives, constants)
        losses_modified_pacanowski_philander = [loss(@view(sol_modified_pacanowski_philander[:,i]), 
                                                     @view(𝒟test_uvT_scaled[:,i])) 
                                                     for i in 1:size(sol_modified_pacanowski_philander, 2)]
        losses_modified_pacanowski_philander_gradient = [loss_gradient(@view(𝒟test_uvT_scaled[:,i]), 
                                                                       @view(sol_modified_pacanowski_philander[:,i]), 
                                                                       @view(𝒟test_uvT_scaled_gradient[:,i]), 
                                                                       @view(sol_modified_pacanowski_philander_gradient[:,i]), 
                                                                       gradient_scaling) for i in 1:size(sol_modified_pacanowski_philander, 2)]
        output["losses_modified_pacanowski_philander"] = losses_modified_pacanowski_philander
        output["loss_modified_pacanowski_philander"] = mean(losses_modified_pacanowski_philander)
        output["losses_modified_pacanowski_philander_gradient"] = losses_modified_pacanowski_philander_gradient .- losses_modified_pacanowski_philander
        output["loss_modified_pacanowski_philander_gradient"] = mean(losses_modified_pacanowski_philander_gradient)
    end

    truth_uw = 𝒟test.uw.coarse[:,trange]
    truth_vw = 𝒟test.vw.coarse[:,trange]
    truth_wT = 𝒟test.wT.coarse[:,trange]
    
    truth_u = 𝒟test.uvT_unscaled[1:Nz, trange]
    truth_v = 𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]
    truth_T = 𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange]

    test_uw = similar(truth_uw)
    test_vw = similar(truth_vw)
    test_wT = similar(truth_wT)

    for i in 1:size(test_uw, 2)
        test_uw[:,i], test_vw[:,i], test_wT[:,i] = predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
    end

    test_uw .= inv(scalings.uw).(test_uw)
    test_vw .= inv(scalings.vw).(test_vw)
    test_wT .= inv(scalings.wT).(test_wT)
    test_u = inv(scalings.u).(sol[1:Nz,:])
    test_v = inv(scalings.v).(sol[Nz + 1:2Nz, :])
    test_T = inv(scalings.T).(sol[2Nz + 1: 3Nz, :])

    depth_profile = 𝒟test.u.z
    depth_flux = 𝒟test.uw.z
    t = 𝒟test.t[trange]

    truth_Ri = similar(𝒟test.uw.coarse[:,trange])

    for i in 1:size(truth_Ri, 2)
        truth_Ri[:,i] .= local_richardson.(D_face * 𝒟test.u.scaled[:,i], D_face * 𝒟test.v.scaled[:,i], D_face * 𝒟test.T.scaled[:,i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end

    test_Ri = similar(truth_Ri)

    for i in 1:size(test_Ri,2)
        test_Ri[:,i] .= local_richardson.(D_face * sol[1:Nz,i], D_face * sol[Nz + 1:2Nz, i], D_face * sol[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end

    output["truth_Ri"] = truth_Ri
    output["test_Ri"] = test_Ri

    if modified_pacanowski_philander
        test_uw_modified_pacanowski_philander = similar(truth_uw)
        test_vw_modified_pacanowski_philander = similar(truth_vw)
        test_wT_modified_pacanowski_philander = similar(truth_wT)

        for i in 1:size(test_uw_modified_pacanowski_philander, 2)
            test_uw_modified_pacanowski_philander[:,i], test_vw_modified_pacanowski_philander[:,i], test_wT_modified_pacanowski_philander[:,i] = 
                                    predict_flux(NN_constructions.uw(zeros(Float32, NN_sizes.uw)), 
                                                NN_constructions.vw(zeros(Float32, NN_sizes.vw)), 
                                                NN_constructions.wT(zeros(Float32, NN_sizes.wT)), 
                                     @view(sol_modified_pacanowski_philander[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
        end

        test_uw_modified_pacanowski_philander .= inv(scalings.uw).(test_uw_modified_pacanowski_philander)
        test_vw_modified_pacanowski_philander .= inv(scalings.vw).(test_vw_modified_pacanowski_philander)
        test_wT_modified_pacanowski_philander .= inv(scalings.wT).(test_wT_modified_pacanowski_philander)
        test_u_modified_pacanowski_philander = inv(scalings.u).(sol_modified_pacanowski_philander[1:Nz,:])
        test_v_modified_pacanowski_philander = inv(scalings.v).(sol_modified_pacanowski_philander[Nz + 1:2Nz, :])
        test_T_modified_pacanowski_philander = inv(scalings.T).(sol_modified_pacanowski_philander[2Nz + 1: 3Nz, :])

        test_Ri_modified_pacanowski_philander = similar(truth_Ri)

        for i in 1:size(test_Ri_modified_pacanowski_philander,2)
            test_Ri_modified_pacanowski_philander[:,i] .= 
            local_richardson.(D_face * sol_modified_pacanowski_philander[1:Nz,i], 
                            D_face * sol_modified_pacanowski_philander[Nz + 1:2Nz, i], 
                            D_face * sol_modified_pacanowski_philander[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
        end

        test_uw_NN_only = similar(truth_uw)
        test_vw_NN_only = similar(truth_vw)
        test_wT_NN_only = similar(truth_wT)

        for i in 1:size(test_uw_NN_only, 2)
            test_uw_NN_only[:,i], test_vw_NN_only[:,i], test_wT_NN_only[:,i] = 
            predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), BCs, conditions, scalings, constants_NN_only, derivatives, filters)
        end

        test_uw_NN_only .= inv(scalings.uw).(test_uw_NN_only)
        test_vw_NN_only .= inv(scalings.vw).(test_vw_NN_only)
        test_wT_NN_only .= inv(scalings.wT).(test_wT_NN_only)

        output["test_Ri_modified_pacanowski_philander"] = test_Ri_modified_pacanowski_philander
    end

    if !unscale
        truth_uw .= scalings.uw.(𝒟test.uw.coarse[:,trange])
        truth_vw .= scalings.vw.(𝒟test.vw.coarse[:,trange])
        truth_wT .= scalings.wT.(𝒟test.wT.coarse[:,trange])

        truth_u .= scalings.u.(𝒟test.uvT_unscaled[1:Nz, trange])
        truth_v .= scalings.v.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange])
        truth_T .= scalings.T.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])

        test_uw .= scalings.uw.(test_uw)
        test_vw .= scalings.vw.(test_vw)
        test_wT .= scalings.wT.(test_wT)

        test_u .= scalings.u.(test_u)
        test_v .= scalings.v.(test_v)
        test_T .= scalings.w.(test_T)

        if modified_pacanowski_philander
            test_uw_modified_pacanowski_philander .= scalings.uw.(test_uw_modified_pacanowski_philander)
            test_vw_modified_pacanowski_philander .= scalings.vw.(test_vw_modified_pacanowski_philander)
            test_wT_modified_pacanowski_philander .= scalings.wT.(test_wT_modified_pacanowski_philander)
    
            test_u_modified_pacanowski_philander .= scalings.u.(test_u_modified_pacanowski_philander)
            test_v_modified_pacanowski_philander .= scalings.v.(test_v_modified_pacanowski_philander)
            test_T_modified_pacanowski_philander .= scalings.w.(test_T_modified_pacanowski_philander)

            test_uw_NN_only .= scalings.uw.(test_uw_NN_only)
            test_vw_NN_only .= scalings.vw.(test_vw_NN_only)
            test_wT_NN_only .= scalings.wT.(test_wT_NN_only)
        end
    end

    if unscale
        test_uw .= test_uw .- test_uw[1, 1]
        test_vw .= test_vw .- test_vw[1, 1] 
        test_wT .= test_wT .- test_wT[1, 1]

        if modified_pacanowski_philander
            test_uw_modified_pacanowski_philander .= test_uw_modified_pacanowski_philander .- test_uw_modified_pacanowski_philander[1, 1]
            test_vw_modified_pacanowski_philander .= test_vw_modified_pacanowski_philander .- test_vw_modified_pacanowski_philander[1, 1] 
            test_wT_modified_pacanowski_philander .= test_wT_modified_pacanowski_philander .- test_wT_modified_pacanowski_philander[1, 1]

            test_uw_NN_only .= test_uw_NN_only .- test_uw_NN_only[1, 1]
            test_vw_NN_only .= test_vw_NN_only .- test_vw_NN_only[1, 1] 
            test_wT_NN_only .= test_wT_NN_only .- test_wT_NN_only[1, 1]
        end
    end

    output["truth_uw"] = truth_uw
    output["truth_vw"] = truth_vw
    output["truth_wT"] = truth_wT

    output["truth_u"] = truth_u
    output["truth_v"] = truth_v
    output["truth_T"] = truth_T

    output["test_uw"] = test_uw
    output["test_vw"] = test_vw
    output["test_wT"] = test_wT

    output["test_u"] = test_u
    output["test_v"] = test_v
    output["test_T"] = test_T

    output["depth_profile"] = 𝒟test.u.z
    output["depth_flux"] = 𝒟test.uw.z
    output["t"] = 𝒟test.t[trange]

    if modified_pacanowski_philander
        output["test_uw_modified_pacanowski_philander"] = test_uw_modified_pacanowski_philander
        output["test_vw_modified_pacanowski_philander"] = test_vw_modified_pacanowski_philander
        output["test_wT_modified_pacanowski_philander"] = test_wT_modified_pacanowski_philander
    
        output["test_u_modified_pacanowski_philander"] = test_u_modified_pacanowski_philander
        output["test_v_modified_pacanowski_philander"] = test_v_modified_pacanowski_philander
        output["test_T_modified_pacanowski_philander"] = test_T_modified_pacanowski_philander

        output["test_uw_NN_only"] = test_uw_NN_only
        output["test_vw_NN_only"] = test_vw_NN_only
        output["test_wT_NN_only"] = test_wT_NN_only
    end

    if OUTPUT_PATH !== ""
        jldopen(OUTPUT_PATH, "w") do file
            file["NDE_profile"] = output
        end
    end

    return output
end

function solve_oceananigans_modified_pacanowski_philander_nn(test_files, EXTRACTED_FILE_PATH, OUTPUT_DIR; timestep=60)
    @info "Loading Training Data..."
    extracted_training_file = jldopen(EXTRACTED_FILE_PATH, "r")

    uw_NN = extracted_training_file["neural_network/uw"]
    vw_NN = extracted_training_file["neural_network/vw"]
    wT_NN = extracted_training_file["neural_network/wT"]

    train_files = extracted_training_file["training_info/train_files"]
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]

    scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling)
    diffusivity_params = extracted_training_file["training_info/parameters"]

    if !ispath(OUTPUT_DIR)
        mkdir(OUTPUT_DIR)
    end

    for test_file in test_files
        @info "Starting $test_file"
        ds = jldopen(directories[test_file])

        f = ds["parameters/coriolis_parameter"]
        α = ds["parameters/thermal_expansion_coefficient"]
        g = ds["parameters/gravitational_acceleration"]
        Nz = 32
        Lz = ds["grid/Lz"]
        Δz = ds["grid/Δz"]

        frames = keys(ds["timeseries/t"])
        stop_time = ds["timeseries/t/$(frames[end])"]

        uw_flux = ds["parameters/boundary_condition_u_top"]
        vw_flux = 0
        wT_flux = ds["parameters/boundary_condition_θ_top"]

        T₀ = Array(ds["timeseries/T/0"][1, 1, :])

        ∂u₀∂z = ds["parameters/boundary_condition_u_bottom"]
        ∂v₀∂z = ds["parameters/boundary_condition_u_bottom"]

        constants = (; f, α, g, Nz, Lz, T₀)
        BCs = (top=(uw=uw_flux, vw=vw_flux, wT=wT_flux), bottom=(u=∂u₀∂z, v=∂v₀∂z))

        if test_file in train_files
            dir_str = "train_$test_file"
        else
            dir_str = "test_$test_file"
        end

        DIR_PATH = joinpath(OUTPUT_DIR, dir_str)

        if !ispath(DIR_PATH)
            mkdir(DIR_PATH)
        end

        BASELINE_RESULTS_PATH = joinpath(DIR_PATH, "baseline_oceananigans")
        NN_RESULTS_PATH = joinpath(DIR_PATH, "NN_oceananigans")

        oceananigans_modified_pacanowski_philander_nn(uw_NN, vw_NN, wT_NN, constants, BCs, scalings, diffusivity_params, 
                                                    BASELINE_RESULTS_PATH=BASELINE_RESULTS_PATH,
                                                    NN_RESULTS_PATH=NN_RESULTS_PATH,
                                                    stop_time=stop_time, Δt=timestep)
    end
end

function NDE_profile_oceananigans(FILE_DIR, train_files, test_files;
                                  ν₀=1f-1, ν₋=1f-4, ΔRi=1f-1, Riᶜ=0.25f0, Pr=1, gradient_scaling,
                                  OUTPUT_PATH="")
    @assert length(test_files) == 1
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
    𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

    BASELINE_SOL_PATH = joinpath(FILE_DIR, "baseline_oceananigans.jld2")
    NDE_SOL_PATH = joinpath(FILE_DIR, "NN_oceananigans.jld2")

    baseline_sol = jldopen(BASELINE_SOL_PATH)
    NDE_sol = jldopen(NDE_SOL_PATH)

    frames = keys(baseline_sol["timeseries/t"])

    @assert length(frames) == length(𝒟test.t)

    Nz = baseline_sol["grid/Nz"]
    α = baseline_sol["buoyancy/model/equation_of_state/α"]
    g = baseline_sol["buoyancy/model/gravitational_acceleration"]
    constants = (; Nz, α, g)
    train_parameters = (ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, gradient_scaling=gradient_scaling)
    derivatives_dimensionless = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))

    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]

    scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling)

    t = 𝒟test.t
    zC = baseline_sol["grid/zC"][2:end-1]
    zF = baseline_sol["grid/zF"][2:end-1]

    truth_u = 𝒟test.u.coarse
    truth_v = 𝒟test.v.coarse
    truth_T = 𝒟test.T.coarse
    
    truth_uw = 𝒟test.uw.coarse
    truth_vw = 𝒟test.vw.coarse
    truth_wT = 𝒟test.wT.coarse

    test_u_modified_pacanowski_philander = similar(truth_u)
    test_v_modified_pacanowski_philander = similar(truth_u)
    test_T_modified_pacanowski_philander = similar(truth_u)

    test_uw_modified_pacanowski_philander = similar(truth_uw)
    test_vw_modified_pacanowski_philander = similar(truth_uw)
    test_wT_modified_pacanowski_philander = similar(truth_uw)

    test_u = similar(truth_u)
    test_v = similar(truth_u)
    test_T = similar(truth_u)

    test_uw = similar(truth_uw)
    test_vw = similar(truth_uw)
    test_wT = similar(truth_uw)

    for i in 1:size(truth_u,2)
        test_u_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/u/$(frames[i])"][:]
        test_v_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/v/$(frames[i])"][:]
        test_T_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/T/$(frames[i])"][:]
        test_uw_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/uw/$(frames[i])"][:]
        test_vw_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/vw/$(frames[i])"][:]
        test_wT_modified_pacanowski_philander[:,i] .= baseline_sol["timeseries/wT/$(frames[i])"][:]

        test_u[:,i] .= NDE_sol["timeseries/u/$(frames[i])"][:]
        test_v[:,i] .= NDE_sol["timeseries/v/$(frames[i])"][:]
        test_T[:,i] .= NDE_sol["timeseries/T/$(frames[i])"][:]
        test_uw[:,i] .= NDE_sol["timeseries/uw/$(frames[i])"][:]
        test_vw[:,i] .= NDE_sol["timeseries/vw/$(frames[i])"][:]
        test_wT[:,i] .= NDE_sol["timeseries/wT/$(frames[i])"][:]
    end
   
    close(baseline_sol)
    close(NDE_sol)

    test_uw_NN_only = similar(truth_uw)
    test_vw_NN_only = similar(truth_uw)
    test_wT_NN_only = similar(truth_uw)

    for i in 1:size(test_uw_NN_only,2)
        uw_total = @view test_uw[:, i]
        vw_total = @view test_vw[:, i]
        wT_total = @view test_wT[:, i]

        uw_modified_pacanowski_philander = @view test_uw_modified_pacanowski_philander[:, i]
        vw_modified_pacanowski_philander = @view test_vw_modified_pacanowski_philander[:, i]
        wT_modified_pacanowski_philander = @view test_wT_modified_pacanowski_philander[:, i]

        test_uw_NN_only[:, i] .= uw_total .+ uw_modified_pacanowski_philander
        test_vw_NN_only[:, i] .= vw_total .+ vw_modified_pacanowski_philander
        test_wT_NN_only[:, i] .= wT_total .+ wT_modified_pacanowski_philander
    end

    D_face = Float32.(Dᶠ(Nz, zC[2] - zC[1]))

    @inline function ∂_∂z(profile)
        output = zeros(typeof(profile[1]), size(profile, 1) + 1, size(profile,2))
        for i in 1:size(profile,2)
            profile_col = @view profile[:,i]
            output_col = @view output[:,i]
            output_col .= D_face * profile_col
        end
        return output
    end

    @inline function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, g, α)
        Bz = g * α * ∂T∂z
        S² = ∂u∂z ^2 + ∂v∂z ^2
        return Bz / S²
    end

    truth_Ri = local_richardson.(∂_∂z(truth_u), ∂_∂z(truth_v), ∂_∂z(truth_T), g, α)
    test_Ri = local_richardson.(∂_∂z(test_u), ∂_∂z(test_v), ∂_∂z(test_T), g, α)
    test_Ri_modified_pacanowski_philander = local_richardson.(∂_∂z(test_u_modified_pacanowski_philander), ∂_∂z(test_v_modified_pacanowski_philander), ∂_∂z(test_T_modified_pacanowski_philander), g, α)

    truth_uvT_scaled = [scalings.u.(𝒟test.uvT_unscaled[1:Nz, :]); 
                        scalings.v.(𝒟test.uvT_unscaled[Nz + 1:2Nz, :]); 
                        scalings.T.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, :])]

    baseline_uvT_scaled = [scalings.u.(test_u_modified_pacanowski_philander); 
                        scalings.v.(test_v_modified_pacanowski_philander); 
                        scalings.T.(test_T_modified_pacanowski_philander)]

    NN_uvT_scaled = [scalings.u.(test_u); 
                     scalings.v.(test_v); 
                     scalings.T.(test_T)]


    ∂z_truth_uvT_scaled = calculate_profile_gradient(truth_uvT_scaled, derivatives_dimensionless, constants)
    ∂z_baseline_uvT_scaled = calculate_profile_gradient(baseline_uvT_scaled, derivatives_dimensionless, constants)
    ∂z_NN_uvT_scaled = calculate_profile_gradient(NN_uvT_scaled, derivatives_dimensionless, constants)
    
    losses = [loss(@view(truth_uvT_scaled[:,i]), @view(NN_uvT_scaled[:,i])) for i in 1:size(truth_uvT_scaled, 2)]

    losses_gradient = [loss_gradient(@view(truth_uvT_scaled[:,i]), 
                                     @view(NN_uvT_scaled[:,i]), 
                                     @view(∂z_truth_uvT_scaled[:,i]), 
                                     @view(∂z_NN_uvT_scaled[:,i]), 
                                     gradient_scaling) for i in 1:size(truth_uvT_scaled, 2)]

    profile_loss = mean(losses)
    profile_loss_gradient = mean(losses_gradient)

    losses_modified_pacanowski_philander = [loss(@view(truth_uvT_scaled[:,i]), 
                                                    @view(baseline_uvT_scaled[:,i])) 
                                                    for i in 1:size(truth_uvT_scaled, 2)]

    losses_modified_pacanowski_philander_gradient = [loss_gradient(@view(truth_uvT_scaled[:,i]), 
                                                                    @view(baseline_uvT_scaled[:,i]), 
                                                                    @view(∂z_truth_uvT_scaled[:,i]), 
                                                                    @view(∂z_baseline_uvT_scaled[:,i]), 
                                                                    gradient_scaling) for i in 1:size(truth_uvT_scaled, 2)]

    profile_loss_modified_pacanowski_philander = mean(losses_modified_pacanowski_philander)
    profile_loss_modified_pacanowski_philander_gradient = mean(losses_modified_pacanowski_philander_gradient)

    output = Dict(
           "depth_profile" => zC,
              "depth_flux" => zF,
                       "t" => t,
        "train_parameters" => train_parameters,

        "truth_u" => truth_u,
        "truth_v" => truth_v,
        "truth_T" => truth_T,
    
        "test_u" => test_u,
        "test_v" => test_v,
        "test_T" => test_T,
    
        "test_u_modified_pacanowski_philander" => test_u_modified_pacanowski_philander,
        "test_v_modified_pacanowski_philander" => test_v_modified_pacanowski_philander,
        "test_T_modified_pacanowski_philander" => test_T_modified_pacanowski_philander,

        "truth_uw" => truth_uw,
        "truth_vw" => truth_vw,
        "truth_wT" => truth_wT,
        
        "test_uw" => test_uw,
        "test_vw" => test_vw,
        "test_wT" => test_wT,
    
        "test_uw_modified_pacanowski_philander" => test_uw_modified_pacanowski_philander,
        "test_vw_modified_pacanowski_philander" => test_vw_modified_pacanowski_philander,
        "test_wT_modified_pacanowski_philander" => test_wT_modified_pacanowski_philander,
    
        "test_uw_NN_only" => test_uw_NN_only,
        "test_vw_NN_only" => test_vw_NN_only,
        "test_wT_NN_only" => test_wT_NN_only,

                                     "truth_Ri" => truth_Ri,
                                      "test_Ri" => test_Ri,
        "test_Ri_modified_pacanowski_philander" => test_Ri_modified_pacanowski_philander,

                                               "losses" => losses,
                                                 "loss" => profile_loss,
                                      "losses_gradient" => losses_gradient,
                                        "loss_gradient" => profile_loss_gradient,
                 "losses_modified_pacanowski_philander" => losses_modified_pacanowski_philander,
                   "loss_modified_pacanowski_philander" => profile_loss_modified_pacanowski_philander,
        "losses_modified_pacanowski_philander_gradient" => losses_modified_pacanowski_philander_gradient,
          "loss_modified_pacanowski_philander_gradient" => profile_loss_modified_pacanowski_philander_gradient,
    )
    
    if OUTPUT_PATH !== ""
        jldopen(OUTPUT_PATH, "w") do file
            file["NDE_profile"] = output
        end
    end

    return output
end


