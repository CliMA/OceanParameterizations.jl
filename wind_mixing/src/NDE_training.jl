function prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    H = abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1])
    τ = abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1])
    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]

    uw_weights, re_uw = Flux.destructure(uw_NN)
    vw_weights, re_vw = Flux.destructure(vw_NN)
    wT_weights, re_wT = Flux.destructure(wT_NN)

    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)

    uw_range = 1:size_uw_NN
    vw_range = size_uw_NN + 1:size_uw_NN + size_vw_NN
    wT_range = size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN

    if conditions.modified_pacanowski_philander
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, ν₀=ν₀, ν₋=ν₋, Riᶜ=Riᶜ, ΔRi=ΔRi, Pr=Pr)
    elseif conditions.convective_adjustment
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, κ=κ)
    else
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α)
    end
    scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling)
    derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))
    NN_constructions = (uw=re_uw, vw=re_vw, wT=re_wT)
    weights = Float32[uw_weights; vw_weights; wT_weights]

    NN_sizes = (uw=size_uw_NN, vw=size_vw_NN, wT=size_wT_NN)
    NN_ranges = (uw=uw_range, vw=vw_range, wT=wT_range)

    filters = (cell=WindMixing.smoothing_filter(Nz, 3), face=WindMixing.smoothing_filter(Nz+1, 3), interior=WindMixing.smoothing_filter(Nz-1, 3))
    return constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters
end

function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, H, g, α, σ_u, σ_v, σ_T)
    # H, g, α = constants.H, constants.g, constants.α
    # σ_u, σ_v, σ_T = scalings.u.σ, scalings.v.σ, scalings.T.σ
    Bz = H * g * α * σ_T * ∂T∂z
    S² = (σ_u * ∂u∂z) ^2 + (σ_v * ∂v∂z) ^2
    return Bz / S²
end

tanh_step(x) = (1 - tanh(x)) / 2

function NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)
    uw_range, vw_range, wT_range = NN_ranges.uw, NN_ranges.vw, NN_ranges.wT
    uw_weights, vw_weights, wT_weights = p[uw_range], p[vw_range], p[wT_range]
    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = p[wT_range[end] + 1:end]
    BCs = (uw=(top=uw_top, bottom=uw_bottom), vw=(top=vw_top, bottom=vw_bottom), wT=(top=wT_top, bottom=wT_bottom))
    re_uw, re_vw, re_wT = NN_constructions.uw, NN_constructions.vw, NN_constructions.wT
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    return predict_NDE(uw_NN, vw_NN, wT_NN, x, BCs, conditions, scalings, constants, derivatives, filters)
end

function predict_flux(uw_NN, vw_NN, wT_NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    Nz, H, τ, f = constants.Nz, constants.H, constants.τ, constants.f
    uw_scaling, vw_scaling, wT_scaling = scalings.uw, scalings.vw, scalings.wT
    σ_uw, σ_vw, σ_wT = uw_scaling.σ, vw_scaling.σ, wT_scaling.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ
    D_cell, D_face = derivatives.cell, derivatives.face

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    uw_interior = uw_NN(x)
    vw_interior = vw_NN(x)
    wT_interior = wT_NN(x)

    if conditions.smooth_NN
        uw_interior = filters.interior * uw_interior
        vw_interior = filters.interior * vw_interior
        wT_interior = filters.interior * wT_interior
    end
    
    if conditions.zero_weights
        uw = [0f0; uw_interior; 0f0]
        vw = [0f0; vw_interior; 0f0]
        wT = [0f0; wT_interior; 0f0]
    else
        uw = [BCs.uw.bottom; uw_interior; BCs.uw.top]
        vw = [BCs.vw.bottom; vw_interior; BCs.vw.top]
        wT = [BCs.wT.bottom; wT_interior; BCs.wT.top]
    end

    if conditions.modified_pacanowski_philander
        ϵ = 1f-7
        ∂u∂z = D_face * u
        ∂v∂z = D_face * v
        ∂T∂z = D_face * T
        Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, constants.H, constants.g, constants.α, scalings.u.σ, scalings.v.σ, scalings.T.σ)

        if conditions.smooth_Ri
            Ri = filters.face * Ri
        end

        ν = constants.ν₀ .+ constants.ν₋ .* tanh_step.((Ri .- constants.Riᶜ) ./ constants.ΔRi)
        # ν = constants.ν₀ .+ constants.ν₋ .* tanh_step.((Ri .- constants.Riᶜ) ./ constants.ΔRi) .+ 1f0 .* tanh_step.((Ri .+ constants.Riᶜ) ./ constants.ΔRi)


        if conditions.zero_weights
            ν∂u∂z = [-(BCs.uw.bottom - scalings.uw(0f0)); σ_u / σ_uw / H .* ν[2:end-1] .* ∂u∂z[2:end-1]; -(BCs.uw.top - scalings.uw(0f0))]
            ν∂v∂z = [-(BCs.vw.bottom - scalings.vw(0f0)); σ_v / σ_vw / H .* ν[2:end-1] .* ∂v∂z[2:end-1]; -(BCs.vw.top - scalings.vw(0f0))]
            ν∂T∂z = [-(BCs.wT.bottom - scalings.wT(0f0)); σ_T / σ_wT / H .* ν[2:end-1] ./ constants.Pr .* ∂T∂z[2:end-1]; -(BCs.wT.top - scalings.wT(0f0))]
        else
            ν∂u∂z = σ_u / σ_uw / H .* ν .* ∂u∂z
            ν∂v∂z = σ_v / σ_vw / H .* ν .* ∂v∂z
            ν∂T∂z = σ_T / σ_wT / H .* ν .* ∂T∂z ./ constants.Pr
        end

        return uw .- ν∂u∂z, vw .- ν∂v∂z, wT .- ν∂T∂z
    elseif conditions.convective_adjustment
        ∂T∂z = D_face * T
        κ∂T∂z = σ_T / σ_wT / H .* κ .* min.(0f0, ∂T∂z)
        return uw, vw, wT .- κ∂T∂z
    else
        return uw, vw, wT
    end
end

function predict_uw(NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    Nz, H, τ, f = constants.Nz, constants.H, constants.τ, constants.f
    uw_scaling, vw_scaling, wT_scaling = scalings.uw, scalings.vw, scalings.wT
    σ_uw, σ_vw, σ_wT = uw_scaling.σ, vw_scaling.σ, wT_scaling.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ
    D_cell, D_face = derivatives.cell, derivatives.face

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    interior = NN(x)

    if conditions.smooth_NN
        interior = filters.interior * interior
    end
    
    if conditions.zero_weights
        uw = [0f0; interior; 0f0]
    else
        uw = [BCs.uw.bottom; interior; BCs.uw.top]
    end

    if conditions.modified_pacanowski_philander
        ϵ = 1f-7
        ∂u∂z = D_face * u
        ∂v∂z = D_face * v
        ∂T∂z = D_face * T
        Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, constants.H, constants.g, constants.α, scalings.u.σ, scalings.v.σ, scalings.T.σ)

        if conditions.smooth_Ri
            Ri = filters.face * Ri
        end

        ν = constants.ν₀ .+ constants.ν₋ .* tanh_step.((Ri .- constants.Riᶜ) ./ constants.ΔRi)

        if conditions.zero_weights
            ν∂u∂z = [-(BCs.uw.bottom - scalings.uw(0f0)); σ_u / σ_uw / H .* ν[2:end-1] .* ∂u∂z[2:end-1]; -(BCs.uw.top - scalings.uw(0f0))]
        else
            ν∂u∂z = σ_u / σ_uw / H .* ν .* ∂u∂z
        end

        return uw .- ν∂u∂z
    else
        return uw
    end
end

function predict_vw(NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    Nz, H, τ, f = constants.Nz, constants.H, constants.τ, constants.f
    uw_scaling, vw_scaling, wT_scaling = scalings.uw, scalings.vw, scalings.wT
    σ_uw, σ_vw, σ_wT = uw_scaling.σ, vw_scaling.σ, wT_scaling.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ
    D_cell, D_face = derivatives.cell, derivatives.face

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    interior = NN(x)

    if conditions.smooth_NN
        interior = filters.interior * interior
    end
    
    if conditions.zero_weights
        vw = [0f0; interior; 0f0]
    else
        vw = [BCs.vw.bottom; interior; BCs.vw.top]
    end

    if conditions.modified_pacanowski_philander
        ϵ = 1f-7
        ∂u∂z = D_face * u
        ∂v∂z = D_face * v
        ∂T∂z = D_face * T
        Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, constants.H, constants.g, constants.α, scalings.u.σ, scalings.v.σ, scalings.T.σ)

        if conditions.smooth_Ri
            Ri = filters.face * Ri
        end

        ν = constants.ν₀ .+ constants.ν₋ .* tanh_step.((Ri .- constants.Riᶜ) ./ constants.ΔRi)
        if conditions.zero_weights
            ν∂v∂z = [-(BCs.vw.bottom - scalings.vw(0f0)); σ_v / σ_vw / H .* ν[2:end-1] .* ∂v∂z[2:end-1]; -(BCs.vw.top - scalings.vw(0f0))]
        else
            ν∂v∂z = σ_v / σ_vw / H .* ν .* ∂v∂z
        end

        return vw .- ν∂v∂z
    else
        return vw
    end
end

function predict_wT(NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    Nz, H, τ, f = constants.Nz, constants.H, constants.τ, constants.f
    uw_scaling, vw_scaling, wT_scaling = scalings.uw, scalings.vw, scalings.wT
    σ_uw, σ_vw, σ_wT = uw_scaling.σ, vw_scaling.σ, wT_scaling.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ
    D_cell, D_face = derivatives.cell, derivatives.face

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    interior = NN(x)

    if conditions.smooth_NN
        interior = filters.interior * interior
    end
    
    if conditions.zero_weights
        wT = [0f0; interior; 0f0]
    else
        wT = [BCs.wT.bottom; interior; BCs.wT.top]
    end

    if conditions.modified_pacanowski_philander
        ϵ = 1f-7
        ∂u∂z = D_face * u
        ∂v∂z = D_face * v
        ∂T∂z = D_face * T
        Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, constants.H, constants.g, constants.α, scalings.u.σ, scalings.v.σ, scalings.T.σ)

        if conditions.smooth_Ri
            Ri = filters.face * Ri
        end

        ν = constants.ν₀ .+ constants.ν₋ .* tanh_step.((Ri .- constants.Riᶜ) ./ constants.ΔRi)
        if conditions.zero_weights
            ν∂T∂z = [-(BCs.wT.bottom - scalings.wT(0f0)); σ_T / σ_wT / H .* ν[2:end-1] ./ constants.Pr .* ∂T∂z[2:end-1]; -(BCs.wT.top - scalings.wT(0f0))]
        else
            ν∂T∂z = σ_T / σ_wT / H .* ν .* ∂T∂z ./ constants.Pr
        end

        return wT .- ν∂T∂z
    elseif conditions.convective_adjustment
        ∂T∂z = D_face * T
        κ∂T∂z = σ_T / σ_wT / H .* κ .* min.(0f0, ∂T∂z)
        return wT .- κ∂T∂z
    else
        return wT
    end
end

function predict_NDE(uw_NN, vw_NN, wT_NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    Nz, H, τ, f = constants.Nz, constants.H, constants.τ, constants.f
    σ_uw, σ_vw, σ_wT = scalings.uw.σ, scalings.vw.σ, scalings.wT.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    uw, vw, wT = predict_flux(uw_NN, vw_NN, wT_NN, x, BCs, conditions, scalings, constants, derivatives, filters)
    
    ∂u∂t = -τ / H * σ_uw / σ_u .* derivatives.cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
    ∂v∂t = -τ / H * σ_vw / σ_v .* derivatives.cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
    ∂T∂t = -τ / H * σ_wT / σ_T .* derivatives.cell * wT

    return [∂u∂t; ∂v∂t; ∂T∂t]
end

function train_NDE(uw_NN, vw_NN, wT_NN, train_files, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage; 
                    maxiters=500, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=2f-4, g=9.80665f0, 
                    modified_pacanowski_philander=false, convective_adjustment=false, smooth_profile=false, smooth_NN=false, smooth_Ri=false, train_gradient=false,
                    zero_weights=false, gradient_scaling=5f-3, training_fractions=nothing)
    @assert !modified_pacanowski_philander || !convective_adjustment

    train_parameters = Dict(
                               "ν₀" => ν₀, 
                               "ν₋" => ν₋, 
                              "ΔRi" => ΔRi, 
                              "Riᶜ" => Riᶜ, 
                               "Pr" => Pr, 
                                "κ" => κ,
    "modified_pacanowski_philander" => modified_pacanowski_philander, 
            "convective_adjustment" => convective_adjustment,
                   "smooth_profile" => smooth_profile, 
                        "smooth_NN" => smooth_NN, 
                        "smooth_Ri" => smooth_Ri, 
                   "train_gradient" => train_gradient,
                     "zero_weights" => zero_weights, 
                 "gradient_scaling" => gradient_scaling, 
               "training_fractions" => training_fractions
    )

    if zero_weights
        @assert modified_pacanowski_philander
    end

    @info "Loading training data"

    𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

    @info "Setting up constants"
    
    n_simulations = length(train_files)
    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                    smooth_profile=smooth_profile, smooth_NN=smooth_NN, smooth_Ri=smooth_Ri, 
                    train_gradient=train_gradient, zero_weights=zero_weights)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    D_face = derivatives.face

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    @info "Setting up training data"

    uvT₀s = [𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟train.t[:,1][tsteps]
    uvT_trains = [𝒟train.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)][:, tsteps] for i in 0:n_simulations - 1]

    u_trains, v_trains, T_trains = split_u.(uvT_trains, Nz), split_v.(uvT_trains, Nz), split_T.(uvT_trains, Nz)

    if train_gradient
        u_trains_gradients = [∂_∂z(sol, D_face) for sol in u_trains]
        v_trains_gradients = [∂_∂z(sol, D_face) for sol in v_trains]
        T_trains_gradients = [∂_∂z(sol, D_face) for sol in T_trains]
    end

    @info "Setting up equations and boundary conditions"

    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)

    t_train = t_train ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    BCs = [[𝒟train.uw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.uw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(prob_NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function determine_loss_scalings()
        if training_fractions === nothing
            loss_scalings = (u=1, v=1, T=1, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
        else
            sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
            u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)

            u_loss = mean(loss.(u_trains, u_sols))
            v_loss = mean(loss.(v_trains, v_sols))
            T_loss = mean(loss.(T_trains, T_sols))
            if train_gradient
                u_sols_gradients = [∂_∂z(sol, D_face) for sol in u_sols]
                v_sols_gradients = [∂_∂z(sol, D_face) for sol in v_sols]
                T_sols_gradients = [∂_∂z(sol, D_face) for sol in T_sols]

                ∂u∂z_loss = mean(loss.(u_trains_gradients, u_sols_gradients))
                ∂v∂z_loss = mean(loss.(v_trains_gradients, v_sols_gradients))
                ∂T∂z_loss = mean(loss.(T_trains_gradients, T_sols_gradients))
            else
                ∂u∂z_loss = 0
                ∂v∂z_loss = 0
                ∂T∂z_loss = 0
            end

            losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss)
            loss_scalings = calculate_loss_scalings(losses, training_fractions, train_gradient)
        end
        return loss_scalings
    end

    @info "Determining training scalings"

    loss_scalings = determine_loss_scalings()

    function loss_NDE(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)
        u_loss = mean(loss.(u_trains, u_sols))
        v_loss = mean(loss.(v_trains, v_sols))
        T_loss = mean(loss.(T_trains, T_sols))

        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=0, ∂v∂z=0, ∂T∂z=0)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses, loss_scalings
    end

    function loss_gradient_NDE(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]

        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)

        u_sols_gradients = [∂_∂z(sol, D_face) for sol in u_sols]
        v_sols_gradients = [∂_∂z(sol, D_face) for sol in v_sols]
        T_sols_gradients = [∂_∂z(sol, D_face) for sol in T_sols]

        u_loss = mean(loss.(u_trains, u_sols))
        v_loss = mean(loss.(v_trains, v_sols))
        T_loss = mean(loss.(T_trains, T_sols))
        ∂u∂z_loss = mean(loss.(u_trains_gradients, u_sols_gradients))
        ∂v∂z_loss = mean(loss.(v_trains_gradients, v_sols_gradients))
        ∂T∂z_loss = mean(loss.(T_trains_gradients, T_sols_gradients))
        
        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses, loss_scalings
    end

    @info "Setting up optimization problem"

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient_NDE, GalacticOptim.AutoZygote())
    else
        f_loss = OptimizationFunction(loss_NDE, GalacticOptim.AutoZygote())
    end

    prob_loss = OptimizationProblem(f_loss, @view(weights[:]), BCs)

    @inline function rounded_percentage(num, den)
        return round(num / den * 100, sigdigits=3)
    end

    @info "Writing metadata"

    write_metadata_NDE_training(FILE_PATH, train_files, epochs, tsteps, train_parameters, loss_scalings, train_optimizers, uw_NN, vw_NN, wT_NN)

    @info "Starting Training"
    for i in 1:length(optimizers), epoch in 1:epochs
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                total_loss = args[2]
                losses = args[3]
                loss_scalings = args[4]
                profile_loss = losses.u + losses.v + losses.T
                gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z

                profile_percent = rounded_percentage(profile_loss, total_loss)
                gradient_percent = rounded_percentage(gradient_loss, total_loss)
                u_percent = rounded_percentage(losses.u, total_loss)
                v_percent = rounded_percentage(losses.v, total_loss)
                T_percent = rounded_percentage(losses.T, total_loss)
                ∂u∂z_percent = rounded_percentage(losses.∂u∂z, total_loss)
                ∂v∂z_percent = rounded_percentage(losses.∂v∂z, total_loss)
                ∂T∂z_percent = rounded_percentage(losses.∂T∂z, total_loss)

                @info "loss = $(total_loss): profile $(profile_percent)%, gradient $(gradient_percent)%, u $(u_percent), v $(v_percent)%, T $(T_percent) ∂u∂z $(∂u∂z_percent), ∂v∂z $(∂v∂z_percent)%, ∂T∂z $(∂T∂z_percent)%, stage $stage, opt $i/$(length(optimizers)), epoch $epoch/$epochs, iter $iter/$maxiters"
                write_data_NDE_training(FILE_PATH, losses, loss_scalings,
                                    NN_constructions.uw(args[1][NN_ranges.uw]), 
                                    NN_constructions.vw(args[1][NN_ranges.vw]), 
                                    NN_constructions.wT(args[1][NN_ranges.wT]), 
                                    stage, opt)
            end
            iter += 1
            false
        end
        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        weights .= res.minimizer
    end
    return NN_constructions.uw(weights[NN_ranges.uw]), NN_constructions.vw(weights[NN_ranges.vw]), NN_constructions.wT(weights[NN_ranges.wT])
end

function solve_NDE_nonmutating(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper; 
                                n_simulations, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=1.67f-4, g=9.81f0)
    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=true, convective_adjustment=false, 
                    smooth_profile=false, smooth_NN=false, smooth_Ri=false, 
                    train_gradient=true, zero_weights=true)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    
    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    uvT₀s = [𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟train.t[:,1][tsteps]

    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)

    t_train = t_train ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    BCs = [[𝒟train.uw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.uw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(prob_NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]
    sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        

    return sols
end

function solve_NDE_nonmutating_backprop(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizer; 
                                maxiters, n_simulations, gradient_scaling, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=1.67f-4, g=9.81f0)

    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=true, convective_adjustment=false, 
    smooth_profile=false, smooth_NN=false, smooth_Ri=false, 
    train_gradient=true, zero_weights=true)

    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    uvT₀s = [𝒟train.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟train.t[:,1][tsteps]
    uvT_trains = [𝒟train.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)][:, tsteps] for i in 0:n_simulations - 1]

    D_face = derivatives.face

    uvT_gradients = [calculate_profile_gradient(uvT, derivatives, constants) for uvT in uvT_trains]

    prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)

    t_train = t_train ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    BCs = [[𝒟train.uw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.uw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.vw.scaled[end,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[1,n_steps * i + tsteps[1]],
            𝒟train.wT.scaled[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    prob_NDEs = [ODEProblem(prob_NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function loss_gradient_NDE(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]
        sol_gradients = [calculate_profile_gradient(sol, derivatives, constants) for sol in sols]
        return mean(loss_gradient.(uvT_trains, sols, uvT_gradients, sol_gradients, gradient_scaling))
    end

    f_loss = OptimizationFunction(loss_gradient_NDE, GalacticOptim.AutoZygote())
    prob_loss = OptimizationProblem(f_loss, weights, BCs)

    iter = 1
    function cb(args...)
        @info "NDE, loss = $(args[2]), iteration = $iter/$maxiters"
        iter += 1
        false
    end

    res = solve(prob_loss, optimizer, cb=cb, maxiters=maxiters)
    weights .= res.minimizer
    return NN_constructions.uw(weights[NN_ranges.uw]), NN_constructions.vw(weights[NN_ranges.vw]), NN_constructions.wT(weights[NN_ranges.wT])
end