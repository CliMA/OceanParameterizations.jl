function prepare_parameters_NN_training(𝒟train, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    H = abs(𝒟train.uw.z[end] - 𝒟train.uw.z[1])
    τ = abs(𝒟train.t[:,1][end] - 𝒟train.t[:,1][1])
    u_scaling = 𝒟train.scalings["u"]
    v_scaling = 𝒟train.scalings["v"]
    T_scaling = 𝒟train.scalings["T"]
    uw_scaling = 𝒟train.scalings["uw"]
    vw_scaling = 𝒟train.scalings["vw"]
    wT_scaling = 𝒟train.scalings["wT"]

    if conditions.modified_pacanowski_philander
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, ν₀=ν₀, ν₋=ν₋, Riᶜ=Riᶜ, ΔRi=ΔRi, Pr=Pr)
    elseif conditions.convective_adjustment
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, κ=κ)
    else
        constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α)
    end
    scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling)
    derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))

    filters = (cell=WindMixing.smoothing_filter(Nz, 3), face=WindMixing.smoothing_filter(Nz+1, 3), interior=WindMixing.smoothing_filter(Nz-1, 3))
    return constants, scalings, derivatives, filters
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

function predict_NN(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

function save_NN_weights(weights, FILE_PATH, filename)
    NN_params = Dict(:weights => weights)
    bson(joinpath(FILE_PATH, "$filename.bson"), NN_params)
end

function loss_flux_gradient(NN, )
    
end

function prepare_NN_training_data(𝒟, NN_type, derivatives)
    @inline training_data_BCs(𝒟, i) = (uw=(top=𝒟.uw.scaled[end,i], bottom=𝒟.uw.scaled[1,i]), 
                                       vw=(top=𝒟.vw.scaled[end,i], bottom=𝒟.vw.scaled[1,i]),
                                       wT=(top=𝒟.wT.scaled[end,i], bottom=𝒟.wT.scaled[1,i]))
    @inline training_data_uw(𝒟, i) = ((profile=𝒟.uvT_scaled[:,i], BCs=training_data_BCs(𝒟,i)), (flux=𝒟.uw.scaled[:,i], flux_gradient=calculate_flux_gradient(𝒟.uw.scaled[:,i], derivatives)))
    @inline training_data_vw(𝒟, i) = ((profile=𝒟.uvT_scaled[:,i], BCs=training_data_BCs(𝒟,i)), (flux=𝒟.vw.scaled[:,i], flux_gradient=calculate_flux_gradient(𝒟.vw.scaled[:,i], derivatives)))
    @inline training_data_wT(𝒟, i) = ((profile=𝒟.uvT_scaled[:,i], BCs=training_data_BCs(𝒟,i)), (flux=𝒟.wT.scaled[:,i], flux_gradient=calculate_flux_gradient(𝒟.wT.scaled[:,i], derivatives)))

    if NN_type == "uw"
        data = [training_data_uw(𝒟, i) for i in 1:size(𝒟.uw.scaled, 2)]
    elseif NN_type == "vw"
        data = [training_data_vw(𝒟, i) for i in 1:size(𝒟.vw.scaled, 2)]
    else
        data = [training_data_wT(𝒟, i) for i in 1:size(𝒟.wT.scaled, 2)]
    end
    return shuffle(data)
end

function calculate_flux_gradient(flux, derivatives)
    return derivatives.cell * flux
end

function train_NN(NN, 𝒟train, optimizers, train_epochs, FILE_PATH, NN_type; ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=1.67f-4, g=9.81f0, 
                  modified_pacanowski_philander=false, convective_adjustment=false, smooth_profile=false, smooth_NN=false, smooth_Ri=false, train_gradient=false,
                  zero_weights=false, gradient_scaling=1f-4)
    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                  smooth_profile=smooth_profile, smooth_NN=smooth_NN, smooth_Ri=smooth_Ri, 
                  train_gradient=train_gradient, zero_weights=zero_weights)

    constants, scalings, derivatives, filters = prepare_parameters_NN_training(𝒟train, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    training_data = prepare_NN_training_data(𝒟train, NN_type, derivatives)

    function NN_loss(input, output)
        if NN_type == "uw"
            NN_flux = predict_uw(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        elseif NN_type == "vw"
            NN_flux = predict_vw(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        else
            NN_flux = predict_wT(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        end
        ∂z_NN_flux = calculate_flux_gradient(NN_flux, derivatives)
        # return loss(NN_flux, output.flux)

        return loss(NN_flux, output.flux) + gradient_scaling * loss(output.flux_gradient, ∂z_NN_flux)
    end

    function total_loss(training_data)
        return mean([NN_loss(data[1], data[2]) for data in training_data])
    end

    # loss(x, y) = loss_gradient(x, y, calculate_flux_gradient(x, derivatives), calculate_flux_gradient(y, derivatives), gradient_scaling)

    for i in 1:length(optimizers), epoch in 1:train_epochs[i]
        opt = optimizers[i]
        function cb()
            @info "$NN_type NN, loss = $(total_loss(training_data)), opt $i/$(length(optimizers)), epoch $epoch/$(train_epochs[i])"
        end
        Flux.train!(NN_loss, Flux.params(NN), training_data, opt, cb=Flux.throttle(cb,10))
        write_data_NN_training(FILE_PATH, total_loss(training_data), NN)
    end

    return Flux.destructure(NN)[1]
end

