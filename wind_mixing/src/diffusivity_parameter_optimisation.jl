function DE(x, p, t, derivatives, scalings, constants, BCs)
    ν₀, ν₋, ΔRi, Riᶜ, Pr = p

    Nz, H, τ, f, g, α  = constants.Nz, constants.H, constants.τ, constants.f, constants.g, constants.α
    σ_uw, σ_vw, σ_wT = scalings.uw.σ, scalings.vw.σ, scalings.wT.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ

    D_cell = derivatives.cell
    D_face = derivatives.face

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    ϵ = 1f-7
    ∂u∂z = D_face * u
    ∂v∂z = D_face * v
    ∂T∂z = D_face * T
    Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, H, g, α, σ_u, σ_v, σ_T)

    ν = ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)
    # ν = zeros(Float32, 33)

    ν∂u∂z = [-(BCs.uw.bottom - scalings.uw(0f0)); σ_u / σ_uw / H .* ν[2:end-1] .* ∂u∂z[2:end-1]; -(BCs.uw.top - scalings.uw(0f0))]
    ν∂v∂z = [-(BCs.vw.bottom - scalings.vw(0f0)); σ_v / σ_vw / H .* ν[2:end-1] .* ∂v∂z[2:end-1]; -(BCs.vw.top - scalings.vw(0f0))]
    ν∂T∂z = [-(BCs.wT.bottom - scalings.wT(0f0)); σ_T / σ_wT / H .* ν[2:end-1] ./ Pr .* ∂T∂z[2:end-1]; -(BCs.wT.top - scalings.wT(0f0))]
    
    ∂u∂t = τ / H * σ_uw / σ_u .* derivatives.cell * ν∂u∂z .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
    ∂v∂t = τ / H * σ_vw / σ_v .* derivatives.cell * ν∂v∂z .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
    ∂T∂t = τ / H * σ_wT / σ_T .* derivatives.cell * ν∂T∂z

    return [∂u∂t; ∂v∂t; ∂T∂t]
end

function optimise_modified_pacanowski_philander(train_files, tsteps, timestepper, optimizers, maxiters, FILE_PATH;
                                                n_simulations, ν₀ = 1f-5, ν₋ = 1f-1, ΔRi=0.1f0, Riᶜ=0.25f0, Pr=1f0,
                                                train_gradient=true, gradient_scaling=5f-3,
                                                training_fractions=nothing)
    @info "Loading training data"
    𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    
    @info "Preparing constants and parameters"

    ν₀_scaling = 1 / ν₀
    ν₋_scaling = 1 / ν₋
    ΔRi_scaling = 1 / ΔRi
    Riᶜ_scaling = 1 / Riᶜ
    Pr_scaling = 1 / Pr

    function scale_parameter(parameter, scaling)
        return parameter * scaling
    end

    function unscale_parameter(parameter, scaling)
        return parameter / scaling
    end
    
    function prepare_parameters()
        Nz = length(𝒟.u.z)
        H = abs(𝒟.uw.z[end] - 𝒟.uw.z[1])
        τ = abs(𝒟.t[:,1][end] - 𝒟.t[:,1][1])
        u_scaling = 𝒟.scalings["u"]
        v_scaling = 𝒟.scalings["v"]
        T_scaling = 𝒟.scalings["T"]
        uw_scaling = 𝒟.scalings["uw"]
        vw_scaling = 𝒟.scalings["vw"]
        wT_scaling = 𝒟.scalings["wT"]

        constants = (H=H, τ=τ, Nz=Nz, f=1f-4, α=2f-4, g=9.80665f0)
        scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling, parameters=[ν₀_scaling, ν₋_scaling, ΔRi_scaling, Riᶜ_scaling, Pr_scaling])
        derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))
        parameters = [ν₀, ν₋, ΔRi, Riᶜ, Pr]
        scaled_parameters = scale_parameter.(parameters, scalings.parameters)

        return constants, scalings, derivatives, scaled_parameters
    end
    
    constants, scalings, derivatives, scaled_parameters = prepare_parameters()

    Nz = constants.Nz

    D_face = derivatives.face

    n_steps = Int(length(@view(𝒟.t[:,1])) / n_simulations)

    uvT₀s = [𝒟.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟.t[:,1][tsteps] ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    uvT_trains = [𝒟.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)][:, tsteps] for i in 0:n_simulations - 1]

    u_trains, v_trains, T_trains = split_u.(uvT_trains, Nz), split_v.(uvT_trains, Nz), split_T.(uvT_trains, Nz)

    if train_gradient
        u_trains_gradients = [∂_∂z(sol, D_face) for sol in u_trains]
        v_trains_gradients = [∂_∂z(sol, D_face) for sol in v_trains]
        T_trains_gradients = [∂_∂z(sol, D_face) for sol in T_trains]
    end    

    @info "Setting up BCs"

    @inline function BC(i)
        index = n_steps * i + tsteps[1]
        return (uw=(bottom=𝒟.uw.scaled[1,index], top=𝒟.uw.scaled[end, index]),
                vw=(bottom=𝒟.vw.scaled[1,index], top=𝒟.vw.scaled[end, index]),
                wT=(bottom=𝒟.wT.scaled[1,index], top=𝒟.wT.scaled[end, index]))
    end

    BCs = [BC(i) for i in 0:n_simulations - 1]    

    @info "Setting up differential equations"

    prob_NDEs = [ODEProblem((x, p, t) -> DE(x, p, t, derivatives, scalings, constants, BCs[i]), uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function determine_loss_scalings()
        if training_fractions === nothing
            loss_scalings = (u=1, v=1, T=1, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
        else
            unscaled_parameters = unscale_parameter.(scaled_parameters, scalings.parameters)
            sols = [Array(solve(prob_NDEs[i], timestepper, p=unscaled_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
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

    function loss_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=unscaled_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)
        u_loss = mean(loss.(u_trains, u_sols))
        v_loss = mean(loss.(v_trains, v_sols))
        T_loss = mean(loss.(T_trains, T_sols))

        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=0, ∂v∂z=0, ∂T∂z=0)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses
    end

    function loss_gradient_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=unscaled_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
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

        return sum(scaled_losses), scaled_losses
    end

    @info "Setting up optimization objective"

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient_mpp, GalacticOptim.AutoZygote())
    else
        f_loss = OptimizationFunction(loss_mpp, GalacticOptim.AutoZygote())
    end

    prob_loss = OptimizationProblem(f_loss, scaled_parameters, lb=[0f0, 0f0, 0f0, 0f0, 0f0], ub=[10f0, 10f0, 10f0, 10f0, 10f0])

    @info "Writing metadata"

    write_metadata_modified_pacanowski_philander_optimisation(FILE_PATH, train_files, maxiters, tsteps, unscale_parameter.(scaled_parameters, scalings.parameters), loss_scalings, optimizers)
    
    @info "Beginning Training"
    
    res = 0
    for i in 1:length(optimizers)
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                parameters = args[1]
                unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
                ν₀, ν₋, ΔRi, Riᶜ, Pr = unscaled_parameters
                losses = args[3]
                profile_loss = losses.u + losses.v + losses.T
                gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z
                total_loss = profile_loss + gradient_loss
                @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $(total_loss): profile = $profile_loss, gradient = $gradient_loss (u = $(losses.u), v = $(losses.v), T = $(losses.T)) , ∂u∂z = $(losses.∂u∂z), ∂v∂z = $(losses.∂v∂z), ∂T∂z = $(losses.∂T∂z)), optimizer $i/$(length(optimizers)), iteration = $iter/$maxiters"
                write_data_modified_pacanowski_philander_optimisation(FILE_PATH, losses, unscaled_parameters)
            end
            iter += 1
            false
        end

        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        scaled_parameters .= res.minimizer
    end

    ν₀, ν₋, ΔRi, Riᶜ = unscale_parameter.(scaled_parameters, scalings.parameters)
    @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $loss"
end

function optimise_modified_pacanowski_philander_noν₀(train_files, tsteps, timestepper, optimizers, maxiters, FILE_PATH;
                                                n_simulations, ν₀ = 1f-5, ν₋ = 1f-1, ΔRi=0.1f0, Riᶜ=0.25f0, Pr=1f0,
                                                train_gradient=true, gradient_scaling=5f-3,
                                                training_fractions=nothing)
    @info "Loading training data"
    𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    
    @info "Preparing constants and parameters"

    # ν₀_scaling = 1 / ν₀
    ν₋_scaling = 1 / ν₋
    ΔRi_scaling = 1 / ΔRi
    Riᶜ_scaling = 1 / Riᶜ
    Pr_scaling = 1 / Pr

    function scale_parameter(parameter, scaling)
        return parameter * scaling
    end

    function unscale_parameter(parameter, scaling)
        return parameter / scaling
    end
    
    function prepare_parameters()
        Nz = length(𝒟.u.z)
        H = abs(𝒟.uw.z[end] - 𝒟.uw.z[1])
        τ = abs(𝒟.t[:,1][end] - 𝒟.t[:,1][1])
        u_scaling = 𝒟.scalings["u"]
        v_scaling = 𝒟.scalings["v"]
        T_scaling = 𝒟.scalings["T"]
        uw_scaling = 𝒟.scalings["uw"]
        vw_scaling = 𝒟.scalings["vw"]
        wT_scaling = 𝒟.scalings["wT"]

        constants = (H=H, τ=τ, Nz=Nz, f=1f-4, α=2f-4, g=9.80665f0)
        scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling, parameters=[ν₋_scaling, ΔRi_scaling, Riᶜ_scaling, Pr_scaling])
        derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))
        parameters = [ν₋, ΔRi, Riᶜ, Pr]
        scaled_parameters = scale_parameter.(parameters, scalings.parameters)

        return constants, scalings, derivatives, scaled_parameters
    end
    
    constants, scalings, derivatives, scaled_parameters = prepare_parameters()

    Nz = constants.Nz

    D_face = derivatives.face

    n_steps = Int(length(@view(𝒟.t[:,1])) / n_simulations)

    uvT₀s = [𝒟.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟.t[:,1][tsteps] ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    uvT_trains = [𝒟.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)][:, tsteps] for i in 0:n_simulations - 1]

    u_trains, v_trains, T_trains = split_u.(uvT_trains, Nz), split_v.(uvT_trains, Nz), split_T.(uvT_trains, Nz)

    if train_gradient
        u_trains_gradients = [∂_∂z(sol, D_face) for sol in u_trains]
        v_trains_gradients = [∂_∂z(sol, D_face) for sol in v_trains]
        T_trains_gradients = [∂_∂z(sol, D_face) for sol in T_trains]
    end    

    @info "Setting up BCs"

    @inline function BC(i)
        index = n_steps * i + tsteps[1]
        return (uw=(bottom=𝒟.uw.scaled[1,index], top=𝒟.uw.scaled[end, index]),
                vw=(bottom=𝒟.vw.scaled[1,index], top=𝒟.vw.scaled[end, index]),
                wT=(bottom=𝒟.wT.scaled[1,index], top=𝒟.wT.scaled[end, index]))
    end

    BCs = [BC(i) for i in 0:n_simulations - 1]    

    @info "Setting up differential equations"

    prob_NDEs = [ODEProblem((x, p, t) -> DE(x, p, t, derivatives, scalings, constants, BCs[i]), uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function determine_loss_scalings()
        if training_fractions === nothing
            loss_scalings = (u=1, v=1, T=1, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
        else
            unscaled_parameters = unscale_parameter.(scaled_parameters, scalings.parameters)
            all_parameters = vcat(ν₀, unscaled_parameters)

            sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
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

    function loss_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        all_parameters = vcat(ν₀, unscaled_parameters)

        sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)
        u_loss = mean(loss.(u_trains, u_sols))
        v_loss = mean(loss.(v_trains, v_sols))
        T_loss = mean(loss.(T_trains, T_sols))

        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=0, ∂v∂z=0, ∂T∂z=0)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses
    end

    function loss_gradient_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        all_parameters = vcat(ν₀, unscaled_parameters)

        sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
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

        return sum(scaled_losses), scaled_losses
    end

    @info "Setting up optimization objective"

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient_mpp, GalacticOptim.AutoZygote())
    else
        f_loss = OptimizationFunction(loss_mpp, GalacticOptim.AutoZygote())
    end

    prob_loss = OptimizationProblem(f_loss, scaled_parameters, lb=[0f0, 0f0, 0f0, 0f0], ub=[10f0, 10f0, 10f0, 10f0])

    @info "Writing metadata"

    write_metadata_modified_pacanowski_philander_optimisation(FILE_PATH, train_files, maxiters, tsteps, unscale_parameter.(scaled_parameters, scalings.parameters), loss_scalings, optimizers)
    
    @info "Beginning Training"
    
    res = 0
    for i in 1:length(optimizers)
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                parameters = args[1]
                unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
                # ν₀, ν₋, ΔRi, Riᶜ, Pr = unscaled_parameters
                ν₋, ΔRi, Riᶜ, Pr = unscaled_parameters

                losses = args[3]
                profile_loss = losses.u + losses.v + losses.T
                gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z
                total_loss = profile_loss + gradient_loss
                @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $(total_loss): profile = $profile_loss, gradient = $gradient_loss (u = $(losses.u), v = $(losses.v), T = $(losses.T)) , ∂u∂z = $(losses.∂u∂z), ∂v∂z = $(losses.∂v∂z), ∂T∂z = $(losses.∂T∂z)), optimizer $i/$(length(optimizers)), iteration = $iter/$maxiters"
                write_data_modified_pacanowski_philander_optimisation(FILE_PATH, losses, unscaled_parameters)
            end
            iter += 1
            false
        end

        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        scaled_parameters .= res.minimizer
    end

    # ν₀, ν₋, ΔRi, Riᶜ = unscale_parameter.(scaled_parameters, scalings.parameters)
    ν₋, ΔRi, Riᶜ, Pr = unscale_parameter.(scaled_parameters, scalings.parameters)

    @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $loss"
end

function optimise_modified_pacanowski_philander_noν₀Pr(train_files, tsteps, timestepper, optimizers, maxiters, FILE_PATH;
                                                n_simulations, ν₀ = 1f-5, ν₋ = 1f-1, ΔRi=0.1f0, Riᶜ=0.25f0, Pr=1f0,
                                                train_gradient=true, gradient_scaling=5f-3,
                                                training_fractions=nothing)
    @info "Loading training data"
    𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    
    @info "Preparing constants and parameters"

    # ν₀_scaling = 1 / ν₀
    # Pr_scaling = 1 / Pr

    ν₋_scaling = 1 / ν₋
    ΔRi_scaling = 1 / ΔRi
    Riᶜ_scaling = 1 / Riᶜ

    function scale_parameter(parameter, scaling)
        return parameter * scaling
    end

    function unscale_parameter(parameter, scaling)
        return parameter / scaling
    end
    
    function prepare_parameters()
        Nz = length(𝒟.u.z)
        H = abs(𝒟.uw.z[end] - 𝒟.uw.z[1])
        τ = abs(𝒟.t[:,1][end] - 𝒟.t[:,1][1])
        u_scaling = 𝒟.scalings["u"]
        v_scaling = 𝒟.scalings["v"]
        T_scaling = 𝒟.scalings["T"]
        uw_scaling = 𝒟.scalings["uw"]
        vw_scaling = 𝒟.scalings["vw"]
        wT_scaling = 𝒟.scalings["wT"]

        constants = (H=H, τ=τ, Nz=Nz, f=1f-4, α=2f-4, g=9.80665f0)
        scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling, parameters=[ν₋_scaling, ΔRi_scaling, Riᶜ_scaling])
        derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))
        parameters = [ν₋, ΔRi, Riᶜ]
        scaled_parameters = scale_parameter.(parameters, scalings.parameters)

        return constants, scalings, derivatives, scaled_parameters
    end
    
    constants, scalings, derivatives, scaled_parameters = prepare_parameters()

    Nz = constants.Nz

    D_face = derivatives.face

    n_steps = Int(length(@view(𝒟.t[:,1])) / n_simulations)

    uvT₀s = [𝒟.uvT_scaled[:,n_steps * i + tsteps[1]] for i in 0:n_simulations - 1]
    t_train = 𝒟.t[:,1][tsteps] ./ constants.τ
    tspan_train = (t_train[1], t_train[end])
    uvT_trains = [𝒟.uvT_scaled[:,n_steps * i + 1:n_steps * (i + 1)][:, tsteps] for i in 0:n_simulations - 1]

    u_trains, v_trains, T_trains = split_u.(uvT_trains, Nz), split_v.(uvT_trains, Nz), split_T.(uvT_trains, Nz)

    if train_gradient
        u_trains_gradients = [∂_∂z(sol, D_face) for sol in u_trains]
        v_trains_gradients = [∂_∂z(sol, D_face) for sol in v_trains]
        T_trains_gradients = [∂_∂z(sol, D_face) for sol in T_trains]
    end    

    @info "Setting up BCs"

    @inline function BC(i)
        index = n_steps * i + tsteps[1]
        return (uw=(bottom=𝒟.uw.scaled[1,index], top=𝒟.uw.scaled[end, index]),
                vw=(bottom=𝒟.vw.scaled[1,index], top=𝒟.vw.scaled[end, index]),
                wT=(bottom=𝒟.wT.scaled[1,index], top=𝒟.wT.scaled[end, index]))
    end

    BCs = [BC(i) for i in 0:n_simulations - 1]    

    @info "Setting up differential equations"

    prob_NDEs = [ODEProblem((x, p, t) -> DE(x, p, t, derivatives, scalings, constants, BCs[i]), uvT₀s[i], tspan_train) for i in 1:n_simulations]

    function determine_loss_scalings()
        if training_fractions === nothing
            loss_scalings = (u=1, v=1, T=1, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
        else
            unscaled_parameters = unscale_parameter.(scaled_parameters, scalings.parameters)
            all_parameters = vcat(ν₀, unscaled_parameters, Pr)

            sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
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

    function loss_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        all_parameters = vcat(ν₀, unscaled_parameters, Pr)

        sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)
        u_loss = mean(loss.(u_trains, u_sols))
        v_loss = mean(loss.(v_trains, v_sols))
        T_loss = mean(loss.(T_trains, T_sols))

        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=0, ∂v∂z=0, ∂T∂z=0)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses
    end

    function loss_gradient_mpp(parameters, p)
        unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
        all_parameters = vcat(ν₀, unscaled_parameters)

        sols = [Array(solve(prob_NDEs[i], timestepper, p=all_parameters, reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=t_train)) for i in 1:n_simulations]        
        
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

        return sum(scaled_losses), scaled_losses
    end

    @info "Setting up optimization objective"

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient_mpp, GalacticOptim.AutoZygote())
    else
        f_loss = OptimizationFunction(loss_mpp, GalacticOptim.AutoZygote())
    end

    prob_loss = OptimizationProblem(f_loss, scaled_parameters, lb=[0f0, 0f0, 0f0], ub=[10f0, 10f0, 10f0])

    @info "Writing metadata"

    write_metadata_modified_pacanowski_philander_optimisation(FILE_PATH, train_files, maxiters, tsteps, unscale_parameter.(scaled_parameters, scalings.parameters), loss_scalings, optimizers)
    
    @info "Beginning Training"
    
    res = 0
    for i in 1:length(optimizers)
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                parameters = args[1]
                unscaled_parameters = unscale_parameter.(parameters, scalings.parameters)
                # ν₀, ν₋, ΔRi, Riᶜ, Pr = unscaled_parameters
                ν₋, ΔRi, Riᶜ = unscaled_parameters

                losses = args[3]
                profile_loss = losses.u + losses.v + losses.T
                gradient_loss = losses.∂u∂z + losses.∂v∂z + losses.∂T∂z
                total_loss = profile_loss + gradient_loss
                @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $(total_loss): profile = $profile_loss, gradient = $gradient_loss (u = $(losses.u), v = $(losses.v), T = $(losses.T)) , ∂u∂z = $(losses.∂u∂z), ∂v∂z = $(losses.∂v∂z), ∂T∂z = $(losses.∂T∂z)), optimizer $i/$(length(optimizers)), iteration = $iter/$maxiters"
                write_data_modified_pacanowski_philander_optimisation(FILE_PATH, losses, unscaled_parameters)
            end
            iter += 1
            false
        end

        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        scaled_parameters .= res.minimizer
    end

    ν₋, ΔRi, Riᶜ = unscale_parameter.(scaled_parameters, scalings.parameters)

    @info "ν₀ = $ν₀, ν₋ = $ν₋, ΔRi = $ΔRi, Riᶜ = $Riᶜ, Pr = $Pr, loss = $loss"
end