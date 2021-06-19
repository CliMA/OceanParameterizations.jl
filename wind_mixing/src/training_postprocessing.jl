

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

function NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange;
                              ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=2f-4, g=9.80665f0, f=1f-4,
                              OUTPUT_PATH = "",
                              modified_pacanowski_philander=false, convective_adjustment=false,
                              smooth_NN=false, smooth_Ri=false,
                              zero_weights=false,
                              loss_scalings = (u=1f0, v=1f0, T=1f0, ∂u∂z=5f-3, ∂v∂z=5f-3, ∂T∂z=5f-3),
                              timestepper=ROCK4())
    
    @assert !modified_pacanowski_philander || !convective_adjustment

    @info "Preparing constants"

    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                    smooth_NN=smooth_NN, smooth_Ri=smooth_Ri,
                    zero_weights=zero_weights)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)

    H, τ, f = constants.H, constants.τ, constants.f
    D_face, D_cell = derivatives.face, derivatives.cell

    BCs = prepare_BCs(𝒟test, scalings)
    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = BCs.uw.bottom, BCs.uw.top, BCs.vw.bottom, BCs.vw.top, BCs.wT.bottom, BCs.wT.top

    @info "Setting up differential equations"
    
    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)

    if modified_pacanowski_philander
        constants_NN_only = (H=constants.H, τ=constants.τ, f=constants.f, Nz=constants.Nz, g=constants.g, α=constants.α, ν₀=0f0, ν₋=0f0, Riᶜ=constants.Riᶜ, ΔRi=constants.ΔRi, Pr=constants.Pr)
    end

    t_test = Float32.(𝒟test.t[trange] ./ constants.τ)
    uvT₀ = [scalings.u(𝒟test.uvT_unscaled[1:Nz, 1]); scalings.v(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); scalings.T(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]

    u_𝒟test_scaled = scalings.u.(split_u(𝒟test.uvT_unscaled, Nz))
    v_𝒟test_scaled = scalings.v.(split_v(𝒟test.uvT_unscaled, Nz))
    T_𝒟test_scaled = scalings.T.(split_T(𝒟test.uvT_unscaled, Nz))
    
    ∂u∂z_𝒟test_scaled = ∂_∂z(u_𝒟test_scaled, D_face)
    ∂v∂z_𝒟test_scaled = ∂_∂z(v_𝒟test_scaled, D_face)
    ∂T∂z_𝒟test_scaled = ∂_∂z(T_𝒟test_scaled, D_face)

    @info "Solving NDEs"

    sol = solve_NDE_mutating(uw_NN, vw_NN, wT_NN, scalings, constants, BCs, derivatives, uvT₀, t_test, timestepper)

    u_sol, v_sol, T_sol = split_u(sol, Nz), split_v(sol, Nz), split_T(sol, Nz)

    ∂u∂z_sol = ∂_∂z(u_sol, D_face)
    ∂v∂z_sol = ∂_∂z(v_sol, D_face)
    ∂T∂z_sol = ∂_∂z(T_sol, D_face)

    unscaled_losses = (
        u = loss_per_tstep(u_sol, u_𝒟test_scaled),
        v = loss_per_tstep(v_sol, v_𝒟test_scaled),
        T = loss_per_tstep(T_sol, T_𝒟test_scaled),
        ∂u∂z = loss_per_tstep(∂u∂z_sol, ∂u∂z_𝒟test_scaled),
        ∂v∂z = loss_per_tstep(∂v∂z_sol, ∂v∂z_𝒟test_scaled),
        ∂T∂z = loss_per_tstep(∂T∂z_sol, ∂T∂z_𝒟test_scaled),
    )

    scaled_losses = apply_loss_scalings(unscaled_losses, loss_scalings)

    profile_losses = scaled_losses.u .+ scaled_losses.v .+ scaled_losses.T
    gradient_losses = scaled_losses.∂u∂z .+ scaled_losses.∂v∂z .+ scaled_losses.∂T∂z

    @info "Solving diffusivity-only equations"
    
    if modified_pacanowski_philander
        zeros_uw_NN = NN_constructions.uw(zeros(Float32, NN_sizes.uw))
        zeros_vw_NN = NN_constructions.vw(zeros(Float32, NN_sizes.vw))
        zeros_wT_NN = NN_constructions.wT(zeros(Float32, NN_sizes.wT))
        
        sol_mpp = solve_NDE_mutating(zeros_uw_NN, zeros_vw_NN, zeros_wT_NN, scalings, constants, BCs, derivatives, uvT₀, t_test, timestepper)
        
        u_sol_mpp, v_sol_mpp, T_sol_mpp = split_u(sol_mpp, Nz), split_v(sol_mpp, Nz), split_T(sol_mpp, Nz)
        
        ∂u∂z_sol_mpp = ∂_∂z(u_sol_mpp, D_face)
        ∂v∂z_sol_mpp = ∂_∂z(v_sol_mpp, D_face)
        ∂T∂z_sol_mpp = ∂_∂z(T_sol_mpp, D_face)
        
        unscaled_losses_mpp = (
            u = loss_per_tstep(u_sol_mpp, u_𝒟test_scaled),
            v = loss_per_tstep(v_sol_mpp, v_𝒟test_scaled),
            T = loss_per_tstep(T_sol_mpp, T_𝒟test_scaled),
            ∂u∂z = loss_per_tstep(∂u∂z_sol_mpp, ∂u∂z_𝒟test_scaled),
            ∂v∂z = loss_per_tstep(∂v∂z_sol_mpp, ∂v∂z_𝒟test_scaled),
            ∂T∂z = loss_per_tstep(∂T∂z_sol_mpp, ∂T∂z_𝒟test_scaled),
            )
            
        scaled_losses_mpp = apply_loss_scalings(unscaled_losses_mpp, loss_scalings)
        
        profile_losses_mpp = scaled_losses_mpp.u .+ scaled_losses_mpp.v .+ scaled_losses_mpp.T
        gradient_losses_mpp = scaled_losses_mpp.∂u∂z .+ scaled_losses_mpp.∂v∂z .+ scaled_losses_mpp.∂T∂z
    end
        
    BCs_unscaled = (uw=(top=𝒟test.uw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    vw=(top=𝒟test.vw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    wT=(top=𝒟test.wT.coarse[end, 1], bottom=𝒟test.wT.coarse[1, 1]))
    
    ICs_unscaled = (u=𝒟test.u.coarse[:,1], v=𝒟test.v.coarse[:,1], T=𝒟test.T.coarse[:,1])
    
    t = 𝒟test.t[trange]

    @info "Solving k-profile parameterizations"

    sol_kpp = column_model_1D_kpp(constants, BCs_unscaled, ICs_unscaled, t, OceanTurb.KPP.Parameters())
    
    ∂u∂z_sol_kpp = ∂_∂z(scalings.u.(sol_kpp.U), D_face)
    ∂v∂z_sol_kpp = ∂_∂z(scalings.v.(sol_kpp.V), D_face)
    ∂T∂z_sol_kpp = ∂_∂z(scalings.T.(sol_kpp.T), D_face)
    
    unscaled_losses_kpp = (
        u = loss_per_tstep(scalings.u.(sol_kpp.U), u_𝒟test_scaled),
        v = loss_per_tstep(scalings.v.(sol_kpp.V), v_𝒟test_scaled),
        T = loss_per_tstep(scalings.T.(sol_kpp.T), T_𝒟test_scaled),
        ∂u∂z = loss_per_tstep(∂u∂z_sol_kpp, ∂u∂z_𝒟test_scaled),
        ∂v∂z = loss_per_tstep(∂v∂z_sol_kpp, ∂v∂z_𝒟test_scaled),
        ∂T∂z = loss_per_tstep(∂T∂z_sol_kpp, ∂T∂z_𝒟test_scaled),
        )
        
    scaled_losses_kpp = apply_loss_scalings(unscaled_losses_kpp, loss_scalings)
    
    profile_losses_kpp = scaled_losses_kpp.u .+ scaled_losses_kpp.v .+ scaled_losses_kpp.T
    gradient_losses_kpp = scaled_losses_kpp.∂u∂z .+ scaled_losses_kpp.∂v∂z .+ scaled_losses_kpp.∂T∂z

    @info "Preparing outputs"
                
    truth_u = 𝒟test.uvT_unscaled[1:Nz, trange]
    truth_v = 𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]
    truth_T = 𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange]

    truth_uw = 𝒟test.uw.coarse[:,trange]
    truth_vw = 𝒟test.vw.coarse[:,trange]
    truth_wT = 𝒟test.wT.coarse[:,trange]

    test_u = inv(scalings.u).(sol[1:Nz,:])
    test_v = inv(scalings.v).(sol[Nz + 1:2Nz, :])
    test_T = inv(scalings.T).(sol[2Nz + 1: 3Nz, :])
    
    test_uw = similar(truth_uw)
    test_vw = similar(truth_vw)
    test_wT = similar(truth_wT)
    
    for i in 1:size(test_uw, 2)
        test_uw[:,i], test_vw[:,i], test_wT[:,i] = predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
    end
    
    test_uw .= inv(scalings.uw).(test_uw)
    test_vw .= inv(scalings.vw).(test_vw)
    test_wT .= inv(scalings.wT).(test_wT)

    truth_Ri = similar(𝒟test.uw.coarse[:,trange])
    test_Ri = similar(truth_Ri)
    
    for i in 1:size(truth_Ri, 2)
        truth_Ri[:,i] .= local_richardson.(D_face * 𝒟test.u.scaled[:,i], D_face * 𝒟test.v.scaled[:,i], D_face * 𝒟test.T.scaled[:,i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
        test_Ri[:,i] .= local_richardson.(D_face * sol[1:Nz,i], D_face * sol[Nz + 1:2Nz, i], D_face * sol[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end
    
    if modified_pacanowski_philander
        test_uw_mpp = similar(truth_uw)
        test_vw_mpp = similar(truth_vw)
        test_wT_mpp = similar(truth_wT)
        
        for i in 1:size(test_uw_mpp, 2)
            test_uw_mpp[:,i], test_vw_mpp[:,i], test_wT_mpp[:,i] = 
            predict_flux(NN_constructions.uw(zeros(Float32, NN_sizes.uw)), 
            NN_constructions.vw(zeros(Float32, NN_sizes.vw)), 
            NN_constructions.wT(zeros(Float32, NN_sizes.wT)), 
            @view(sol_mpp[:,i]), BCs, conditions, scalings, constants, derivatives, filters)
        end
        
        test_uw_mpp .= inv(scalings.uw).(test_uw_mpp)
        test_vw_mpp .= inv(scalings.vw).(test_vw_mpp)
        test_wT_mpp .= inv(scalings.wT).(test_wT_mpp)
        test_u_mpp = inv(scalings.u).(sol_mpp[1:Nz,:])
        test_v_mpp = inv(scalings.v).(sol_mpp[Nz + 1:2Nz, :])
        test_T_mpp = inv(scalings.T).(sol_mpp[2Nz + 1: 3Nz, :])
        
        test_Ri_mpp = similar(truth_Ri)
        
        for i in 1:size(test_Ri_mpp,2)
            test_Ri_mpp[:,i] .= 
            local_richardson.(D_face * sol_mpp[1:Nz,i], 
            D_face * sol_mpp[Nz + 1:2Nz, i], 
            D_face * sol_mpp[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
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
    end

    test_u_kpp = sol_kpp.U
    test_v_kpp = sol_kpp.V
    test_T_kpp = sol_kpp.T

    test_uw_kpp = sol_kpp.UW
    test_vw_kpp = sol_kpp.VW
    test_wT_kpp = sol_kpp.WT

    test_Ri_kpp = similar(truth_Ri)
    
    for i in 1:size(test_Ri_kpp,2)
        test_Ri_kpp[:,i] .= local_richardson.(D_face * scalings.u.(@view(sol_kpp.U[:,i])), 
        D_face * scalings.v.(@view(sol_kpp.V[:,i])), 
        D_face * scalings.T.(@view(sol_kpp.T[:,i])), 
        H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
    end
           
    test_uw .= test_uw .- test_uw[1, 1]
    test_vw .= test_vw .- test_vw[1, 1] 
    test_wT .= test_wT .- test_wT[1, 1]
    
    if modified_pacanowski_philander
        test_uw_mpp .= test_uw_mpp .- test_uw_mpp[1, 1]
        test_vw_mpp .= test_vw_mpp .- test_vw_mpp[1, 1] 
        test_wT_mpp .= test_wT_mpp .- test_wT_mpp[1, 1]
        
        test_uw_NN_only .= test_uw_NN_only .- test_uw_NN_only[1, 1]
        test_vw_NN_only .= test_vw_NN_only .- test_vw_NN_only[1, 1] 
        test_wT_NN_only .= test_wT_NN_only .- test_wT_NN_only[1, 1]
    end
    
    depth_profile = 𝒟test.u.z
    depth_flux = 𝒟test.uw.z

    @info "Writing outputs"

    output = Dict()

    output["depth_profile"] = 𝒟test.u.z
    output["depth_flux"] = 𝒟test.uw.z
    output["t"] = 𝒟test.t[trange]

    output["truth_u"] = truth_u
    output["truth_v"] = truth_v
    output["truth_T"] = truth_T
    
    output["truth_uw"] = truth_uw
    output["truth_vw"] = truth_vw
    output["truth_wT"] = truth_wT

    output["truth_Ri"] = truth_Ri
    
    output["test_u"] = test_u
    output["test_v"] = test_v
    output["test_T"] = test_T
    
    output["test_uw"] = test_uw
    output["test_vw"] = test_vw
    output["test_wT"] = test_wT

    output["test_Ri"] = test_Ri
    
    output["u_losses"] = scaled_losses.u
    output["v_losses"] = scaled_losses.v
    output["T_losses"] = scaled_losses.T
    output["∂u∂z_losses"] = scaled_losses.∂u∂z
    output["∂v∂z_losses"] = scaled_losses.∂v∂z
    output["∂T∂z_losses"] = scaled_losses.∂T∂z

    output["losses"] = profile_losses
    output["loss"] = mean(profile_losses)
    output["losses_gradient"] = gradient_losses
    output["loss_gradient"] = mean(gradient_losses)

    if modified_pacanowski_philander
        output["train_parameters"] = (ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, loss_scalings=loss_scalings)
        
        output["test_u_modified_pacanowski_philander"] = test_u_mpp
        output["test_v_modified_pacanowski_philander"] = test_v_mpp
        output["test_T_modified_pacanowski_philander"] = test_T_mpp

        output["test_uw_modified_pacanowski_philander"] = test_uw_mpp
        output["test_vw_modified_pacanowski_philander"] = test_vw_mpp
        output["test_wT_modified_pacanowski_philander"] = test_wT_mpp
                
        output["test_uw_NN_only"] = test_uw_NN_only
        output["test_vw_NN_only"] = test_vw_NN_only
        output["test_wT_NN_only"] = test_wT_NN_only

        output["test_Ri_modified_pacanowski_philander"] = test_Ri_mpp

        output["u_losses_modified_pacanowski_philander"] = scaled_losses_mpp.u
        output["v_losses_modified_pacanowski_philander"] = scaled_losses_mpp.v
        output["T_losses_modified_pacanowski_philander"] = scaled_losses_mpp.T
        output["∂u∂z_losses_modified_pacanowski_philander"] = scaled_losses_mpp.∂u∂z
        output["∂v∂z_losses_modified_pacanowski_philander"] = scaled_losses_mpp.∂v∂z
        output["∂T∂z_losses_modified_pacanowski_philander"] = scaled_losses_mpp.∂T∂z

        output["losses_modified_pacanowski_philander"] = profile_losses_mpp
        output["loss_modified_pacanowski_philander"] = mean(profile_losses_mpp)
        output["losses_modified_pacanowski_philander_gradient"] = gradient_losses_mpp
        output["loss_modified_pacanowski_philander_gradient"] = mean(gradient_losses_mpp)
    end
        
    output["test_u_kpp"] = test_u_kpp
    output["test_v_kpp"] = test_v_kpp
    output["test_T_kpp"] = test_T_kpp
    
    output["test_uw_kpp"] = test_uw_kpp
    output["test_vw_kpp"] = test_vw_kpp
    output["test_wT_kpp"] = test_wT_kpp

    output["test_Ri_kpp"] = test_Ri_kpp

    output["u_losses_kpp"] = scaled_losses_kpp.u
    output["v_losses_kpp"] = scaled_losses_kpp.v
    output["T_losses_kpp"] = scaled_losses_kpp.T
    output["∂u∂z_losses_kpp"] = scaled_losses_kpp.∂u∂z
    output["∂v∂z_losses_kpp"] = scaled_losses_kpp.∂v∂z
    output["∂T∂z_losses_kpp"] = scaled_losses_kpp.∂T∂z

    output["losses_kpp"] = profile_losses_kpp
    output["loss_kpp"] = mean(profile_losses_kpp)
    output["losses_kpp_gradient"] = gradient_losses_kpp
    output["loss_kpp_gradient"] = mean(gradient_losses_kpp)

    if OUTPUT_PATH !== ""
        @info "Saving file"
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
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

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
                                  ν₀=1f-1, ν₋=1f-4, ΔRi=1f-1, Riᶜ=0.25f0, Pr=1, 
                                  loss_scalings=(u=1f0, v=1f0, T=1f0, ∂u∂z=5f-3, ∂v∂z=5f-3, ∂T∂z=5f-3),
                                  OUTPUT_PATH="")
    @assert length(test_files) == 1
    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

    @info "Reading files"

    BASELINE_SOL_PATH = joinpath(FILE_DIR, "baseline_oceananigans.jld2")
    NDE_SOL_PATH = joinpath(FILE_DIR, "NN_oceananigans.jld2")

    baseline_sol = jldopen(BASELINE_SOL_PATH)
    NDE_sol = jldopen(NDE_SOL_PATH)

    frames = keys(baseline_sol["timeseries/t"])

    @assert length(frames) == length(𝒟test.t)

    @info "Loading constants"

    Nz = baseline_sol["grid/Nz"]
    α = baseline_sol["buoyancy/model/equation_of_state/α"]
    g = baseline_sol["buoyancy/model/gravitational_acceleration"]
    constants = (; Nz, α, g)
    train_parameters = (; ν₀, ν₋, ΔRi, Riᶜ, Pr, loss_scalings)

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

    @info "Loading solutions"

    truth_u = 𝒟test.u.coarse
    truth_v = 𝒟test.v.coarse
    truth_T = 𝒟test.T.coarse
    
    truth_uw = 𝒟test.uw.coarse
    truth_vw = 𝒟test.vw.coarse
    truth_wT = 𝒟test.wT.coarse

    test_u_mpp = similar(truth_u)
    test_v_mpp = similar(truth_u)
    test_T_mpp = similar(truth_u)

    test_uw_mpp = similar(truth_uw)
    test_vw_mpp = similar(truth_uw)
    test_wT_mpp = similar(truth_uw)

    test_u = similar(truth_u)
    test_v = similar(truth_u)
    test_T = similar(truth_u)

    test_uw = similar(truth_uw)
    test_vw = similar(truth_uw)
    test_wT = similar(truth_uw)

    for i in 1:size(truth_u,2)
        test_u_mpp[:,i] .= baseline_sol["timeseries/u/$(frames[i])"][:]
        test_v_mpp[:,i] .= baseline_sol["timeseries/v/$(frames[i])"][:]
        test_T_mpp[:,i] .= baseline_sol["timeseries/T/$(frames[i])"][:]
        test_uw_mpp[:,i] .= baseline_sol["timeseries/uw/$(frames[i])"][:]
        test_vw_mpp[:,i] .= baseline_sol["timeseries/vw/$(frames[i])"][:]
        test_wT_mpp[:,i] .= baseline_sol["timeseries/wT/$(frames[i])"][:]

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

        uw_mpp = @view test_uw_mpp[:, i]
        vw_mpp = @view test_vw_mpp[:, i]
        wT_mpp = @view test_wT_mpp[:, i]

        test_uw_NN_only[:, i] .= uw_total .+ uw_mpp
        test_vw_NN_only[:, i] .= vw_total .+ vw_mpp
        test_wT_NN_only[:, i] .= wT_total .+ wT_mpp
    end

    @inline function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, g, α)
        Bz = g * α * ∂T∂z
        S² = ∂u∂z ^2 + ∂v∂z ^2
        return Bz / S²
    end

    D_face = Float32.(Dᶠ(Nz, zC[2] - zC[1]))
    D_face_dimensionless = derivatives_dimensionless.face

    truth_Ri = local_richardson.(∂_∂z(truth_u, D_face), ∂_∂z(truth_v, D_face), ∂_∂z(truth_T, D_face), g, α)
    test_Ri = local_richardson.(∂_∂z(test_u, D_face), ∂_∂z(test_v, D_face), ∂_∂z(test_T, D_face), g, α)
    test_Ri_modified_pacanowski_philander = local_richardson.(∂_∂z(test_u_mpp, D_face), ∂_∂z(test_v_mpp, D_face), ∂_∂z(test_T_mpp, D_face), g, α)

    @info "Calculating Losses"

    truth_u_scaled = scalings.u.(split_u(𝒟test.uvT_unscaled, Nz))
    truth_v_scaled = scalings.v.(split_v(𝒟test.uvT_unscaled, Nz))
    truth_T_scaled = scalings.T.(split_T(𝒟test.uvT_unscaled, Nz))

    baseline_u_scaled = scalings.u.(test_u_mpp)
    baseline_v_scaled = scalings.v.(test_v_mpp)
    baseline_T_scaled = scalings.T.(test_T_mpp)

    test_u_scaled = scalings.u.(test_u)
    test_v_scaled = scalings.v.(test_v)
    test_T_scaled = scalings.T.(test_T)

    truth_∂u∂z_scaled = ∂_∂z(truth_u_scaled, D_face_dimensionless)
    truth_∂v∂z_scaled = ∂_∂z(truth_v_scaled, D_face_dimensionless)
    truth_∂T∂z_scaled = ∂_∂z(truth_T_scaled, D_face_dimensionless)

    baseline_∂u∂z_scaled = ∂_∂z(baseline_u_scaled, D_face_dimensionless)
    baseline_∂v∂z_scaled = ∂_∂z(baseline_v_scaled, D_face_dimensionless)
    baseline_∂T∂z_scaled = ∂_∂z(baseline_T_scaled, D_face_dimensionless)

    test_∂u∂z_scaled = ∂_∂z(test_u_scaled, D_face_dimensionless)
    test_∂v∂z_scaled = ∂_∂z(test_v_scaled, D_face_dimensionless)
    test_∂T∂z_scaled = ∂_∂z(test_T_scaled, D_face_dimensionless)

    u_loss_unscaled = loss_per_tstep(truth_u_scaled, test_u_scaled)
    u_loss_unscaled = loss_per_tstep(truth_u_scaled, test_u_scaled)
    u_loss_unscaled = loss_per_tstep(truth_u_scaled, test_u_scaled)

    u_loss = loss_per_tstep(truth_u_scaled, test_u_scaled)
    u_loss = loss_per_tstep(truth_u_scaled, test_u_scaled)
    u_loss = loss_per_tstep(truth_u_scaled, test_u_scaled)

    unscaled_losses = (
        u = loss_per_tstep(truth_u_scaled, test_u_scaled),
        v = loss_per_tstep(truth_v_scaled, test_v_scaled),
        T = loss_per_tstep(truth_T_scaled, test_T_scaled),
        ∂u∂z = loss_per_tstep(truth_∂u∂z_scaled, test_∂u∂z_scaled),
        ∂v∂z = loss_per_tstep(truth_∂v∂z_scaled, test_∂v∂z_scaled),
        ∂T∂z = loss_per_tstep(truth_∂T∂z_scaled, test_∂T∂z_scaled),
        )

    scaled_losses = apply_loss_scalings(unscaled_losses, loss_scalings)

    profile_losses = scaled_losses.u .+ scaled_losses.v .+ scaled_losses.T
    gradient_losses = scaled_losses.∂u∂z .+ scaled_losses.∂v∂z .+ scaled_losses.∂T∂z

    profile_loss = mean(profile_losses)
    loss_gradient = mean(gradient_losses)

    unscaled_losses_mpp = (
        u = loss_per_tstep(truth_u_scaled, baseline_u_scaled),
        v = loss_per_tstep(truth_v_scaled, baseline_v_scaled),
        T = loss_per_tstep(truth_T_scaled, baseline_T_scaled),
        ∂u∂z = loss_per_tstep(truth_∂u∂z_scaled, baseline_∂u∂z_scaled),
        ∂v∂z = loss_per_tstep(truth_∂v∂z_scaled, baseline_∂v∂z_scaled),
        ∂T∂z = loss_per_tstep(truth_∂T∂z_scaled, baseline_∂T∂z_scaled),
        )

    scaled_losses_mpp = apply_loss_scalings(unscaled_losses_mpp, loss_scalings)

    profile_losses_mpp = scaled_losses_mpp.u .+ scaled_losses_mpp.v .+ scaled_losses_mpp.T
    gradient_losses_mpp = scaled_losses_mpp.∂u∂z .+ scaled_losses_mpp.∂v∂z .+ scaled_losses_mpp.∂T∂z

    profile_loss_mpp = mean(profile_losses_mpp)
    loss_gradient_mpp = mean(gradient_losses_mpp)

    @info "Writing outputs"

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
    
        "test_u_modified_pacanowski_philander" => test_u_mpp,
        "test_v_modified_pacanowski_philander" => test_v_mpp,
        "test_T_modified_pacanowski_philander" => test_T_mpp,

        "truth_uw" => truth_uw,
        "truth_vw" => truth_vw,
        "truth_wT" => truth_wT,
        
        "test_uw" => test_uw,
        "test_vw" => test_vw,
        "test_wT" => test_wT,
    
        "test_uw_modified_pacanowski_philander" => test_uw_mpp,
        "test_vw_modified_pacanowski_philander" => test_vw_mpp,
        "test_wT_modified_pacanowski_philander" => test_wT_mpp,
    
        "test_uw_NN_only" => test_uw_NN_only,
        "test_vw_NN_only" => test_vw_NN_only,
        "test_wT_NN_only" => test_wT_NN_only,

                                     "truth_Ri" => truth_Ri,
                                      "test_Ri" => test_Ri,
        "test_Ri_modified_pacanowski_philander" => test_Ri_modified_pacanowski_philander,

           "u_losses" => scaled_losses.u,
           "v_losses" => scaled_losses.v,
           "T_losses" => scaled_losses.T,
        "∂u∂z_losses" => scaled_losses.∂u∂z,
        "∂v∂z_losses" => scaled_losses.∂v∂z,
        "∂T∂z_losses" => scaled_losses.∂T∂z,

           "u_losses_modified_pacanowski_philander" => scaled_losses_mpp.u,
           "v_losses_modified_pacanowski_philander" => scaled_losses_mpp.v,
           "T_losses_modified_pacanowski_philander" => scaled_losses_mpp.T,
        "∂u∂z_losses_modified_pacanowski_philander" => scaled_losses_mpp.∂u∂z,
        "∂v∂z_losses_modified_pacanowski_philander" => scaled_losses_mpp.∂v∂z,
        "∂T∂z_losses_modified_pacanowski_philander" => scaled_losses_mpp.∂T∂z,

                                               "losses" => scaled_losses.u .+ scaled_losses.v .+ scaled_losses.T,
                                                 "loss" => profile_loss,
                                      "losses_gradient" => scaled_losses.∂u∂z .+ scaled_losses.∂v∂z .+ scaled_losses.∂T∂z,
                                        "loss_gradient" => loss_gradient,
                 "losses_modified_pacanowski_philander" => scaled_losses_mpp.u .+ scaled_losses_mpp.v .+ scaled_losses_mpp.T,
                   "loss_modified_pacanowski_philander" => profile_loss_mpp,
        "losses_modified_pacanowski_philander_gradient" => scaled_losses_mpp.∂u∂z .+ scaled_losses_mpp.∂v∂z .+ scaled_losses_mpp.∂T∂z,
          "loss_modified_pacanowski_philander_gradient" => loss_gradient_mpp,
    )
    
    if OUTPUT_PATH !== ""
        @info "Writing file"
        jldopen(OUTPUT_PATH, "w") do file
            file["NDE_profile"] = output
        end
    end

    return output
end


