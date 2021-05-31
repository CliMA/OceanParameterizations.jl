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
                  zero_weights=false)
    Nz = length(𝒟train.u.z)

    conditions = (modified_pacanowski_philander=modified_pacanowski_philander, convective_adjustment=convective_adjustment, 
                  smooth_profile=smooth_profile, smooth_NN=smooth_NN, smooth_Ri=smooth_Ri, 
                  train_gradient=train_gradient, zero_weights=zero_weights)

    constants, scalings, derivatives, filters = prepare_parameters_NN_training(𝒟train, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    training_data = prepare_NN_training_data(𝒟train, NN_type, derivatives)

    gradient_scaling = 1f-2

    function NN_loss(input, output)
        if NN_type == "uw"
            NN_flux = predict_uw(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        elseif NN_type == "vw"
            NN_flux = predict_vw(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        else
            NN_flux = predict_wT(NN, input.profile, input.BCs, conditions, scalings, constants, derivatives, filters)
        end
        ∂z_NN_flux = calculate_flux_gradient(NN_flux, derivatives)
        return loss_gradient(NN_flux, output.flux, output.flux_gradient, ∂z_NN_flux, gradient_scaling)
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

