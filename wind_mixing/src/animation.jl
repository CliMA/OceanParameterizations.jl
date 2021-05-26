function animate_NN(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str, PATH=joinpath(pwd(), "Output"))
    anim = @animate for n in 1:size(xs[1], 2)
        x_max = maximum(maximum(x) for x in xs)
        x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1], 2))"
        fig = Plots.plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            Plots.plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n] / 86400, digits=2)) days")
        end
        Plots.xlabel!(fig, "$x_str")
        Plots.ylabel!(fig, "z")
    end
    # gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
    mp4(anim, joinpath(PATH, "$(filename).mp4"), fps=30)
end


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
    # return [uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top]
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
        # prob_NDE_NN_only(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants_NN_only, derivatives, filters)
    end


    t_test = Float32.(𝒟test.t[trange] ./ constants.τ)
    tspan_test = (t_test[1], t_test[end])
    uvT₀ = [scalings.u(𝒟test.uvT_unscaled[1:Nz, 1]); scalings.v(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); scalings.T(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]
    prob = ODEProblem(prob_NDE, uvT₀, tspan_test)
    sol = Array(solve(prob, ROCK4(), p=[weights; uw_bottom; uw_top; vw_bottom; vw_top; wT_bottom; wT_top], saveat=t_test))

    if modified_pacanowski_philander
        sol_modified_pacanowski_philander = Array(solve(prob, ROCK4(), p=[zeros(Float32, length(weights)); uw_bottom; uw_top; vw_bottom; vw_top; wT_bottom; wT_top], saveat=t_test))
        
        # prob_NN_only = ODEProblem(prob_NDE_NN_only, uvT₀, tspan_test)
        # sol_NN_only = Array(solve(prob_NN_only, ROCK4(), p=[weights; uw_bottom; uw_top; vw_bottom; vw_top; wT_bottom; wT_top], saveat=t_test))
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
        # test_u_NN_only = inv(scalings.u).(sol_NN_only[1:Nz,:])
        # test_v_NN_only = inv(scalings.v).(sol_NN_only[Nz + 1:2Nz, :])
        # test_T_NN_only = inv(scalings.T).(sol_NN_only[2Nz + 1: 3Nz, :])

        # test_Ri_NN_only = similar(truth_Ri)

        # for i in 1:size(test_Ri_NN_only,2)
        #     test_Ri_NN_only[:,i] .= 
        #     local_richardson.(D_face * sol_NN_only[1:Nz,i], 
        #                     D_face * sol_NN_only[Nz + 1:2Nz, i], 
        #                     D_face * sol_NN_only[2Nz + 1: 3Nz, i], H, g, α, scalings.u.σ, scalings.v.σ, scalings.T.σ)
        # end

        output["test_Ri_modified_pacanowski_philander"] = test_Ri_modified_pacanowski_philander
        # output["test_Ri_NN_only"] = test_Ri_NN_only
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
    
            # test_u_NN_only .= scalings.u.(test_u_NN_only)
            # test_v_NN_only .= scalings.v.(test_v_NN_only)
            # test_T_NN_only .= scalings.w.(test_T_NN_only)

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
    
        # output["test_u_NN_only"] = test_u_NN_only
        # output["test_v_NN_only"] = test_v_NN_only
        # output["test_T_NN_only"] = test_T_NN_only
    end
    return output
end

function NDE_profile_oceananigans(baseline_sol, NDE_sol)
    output = Dict()
    t = parse.(Float64, keys(baseline_sol["timeseries/t"])) .* 60
    zC = baseline_sol["grid/zC"][2:end-1]
    output["depth_profile"] = zC
    # output["depth_flux"] = 𝒟test.uw.z
    output["t"] = t
    
    truth_u = zeros(length(zC), length(t))
    truth_v = similar(truth_u)
    truth_T = similar(truth_u)

    test_u = similar(truth_u)
    test_v = similar(truth_u)
    test_T = similar(truth_u)

    for i in 1:size(truth_u,2)
        truth_u[:,i] .= baseline_sol["timeseries/u/$(keys(baseline_sol["timeseries/t"])[i])"][:]
        truth_v[:,i] .= baseline_sol["timeseries/v/$(keys(baseline_sol["timeseries/t"])[i])"][:]
        truth_T[:,i] .= baseline_sol["timeseries/T/$(keys(baseline_sol["timeseries/t"])[i])"][:]
        test_u[:,i] .= NDE_sol["timeseries/u/$(keys(baseline_sol["timeseries/t"])[i])"][:]
        test_v[:,i] .= NDE_sol["timeseries/v/$(keys(baseline_sol["timeseries/t"])[i])"][:]
        test_T[:,i] .= NDE_sol["timeseries/T/$(keys(baseline_sol["timeseries/t"])[i])"][:]
    end

    output["truth_u"] = truth_u
    output["truth_v"] = truth_v
    output["truth_T"] = truth_T

    output["test_u"] = test_u
    output["test_v"] = test_v
    output["test_T"] = test_T


    return output
end

function animate_profile(data, profile_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_profile = data["truth_$profile_type"]
    test_profile = data["test_$profile_type"]

    profile_max = maximum([maximum(truth_profile), maximum(test_profile)])
    profile_min = minimum([minimum(truth_profile), minimum(test_profile)])

    t = data["t"]

    z_profile = data["depth_profile"]

    z_max = maximum(z_profile)
    z_min = minimum(z_profile)

    anim = @animate for i in 1:length(t)
        @info "Animating $profile_type frame $i/$(length(t))"
        fig = plot(truth_profile[:,i], z_profile, xlim=(profile_min, profile_max), ylim=(z_min, z_max), label="Truth", title="$(round(t[i]/86400, digits=2)) days")
        plot!(fig, test_profile[:,i], z_profile, label="NN")
        ylabel!(fig, "z /m")

        if dimensionless
            xlabel!(fig, profile_type)
        elseif profile_type == "T"
            xlabel!(fig, "T /K")
        else
            xlabel!(fig, "$profile_type /m s⁻¹")
        end

        if i == 1
            savefig(fig, "$FILE_PATH.pdf")
            savefig(fig, "$FILE_PATH.png")
        end
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_flux(data, flux_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_flux = data["truth_$flux_type"]
    test_flux = data["test_$flux_type"]

    flux_max = maximum([maximum(truth_flux), maximum(test_flux)])
    flux_min = minimum([minimum(truth_flux), minimum(test_flux)])

    t = data["t"]

    z_flux = data["depth_flux"]

    z_max = maximum(z_flux)
    z_min = minimum(z_flux)

    anim = @animate for i in 1:length(t)
        @info "Animating $flux_type frame $i/$(length(t))"
        fig = plot(truth_flux[:,i], z_flux, xlim=(flux_min, flux_max), ylim=(z_min, z_max), label="Truth", title="$(round(t[i]/86400, digits=2)) days")
        plot!(fig, test_flux[:,i], z_flux, label = "NN")
        ylabel!(fig, "z /m")

        if dimensionless
            xlabel!(fig, flux_type)
        else
            xlabel!(fig, "$flux_type /m² s⁻²")
        end

    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profile_flux(data, profile_type, flux_type, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_flux = data["truth_$flux_type"]
    test_flux = data["test_$flux_type"]

    truth_profile = data["truth_$profile_type"]
    test_profile = data["test_$profile_type"]

    flux_max = maximum([maximum(truth_flux), maximum(test_flux)])
    flux_min = minimum([minimum(truth_flux), minimum(test_flux)])

    profile_max = maximum([maximum(truth_profile), maximum(test_profile)])
    profile_min = minimum([minimum(truth_profile), minimum(test_profile)])

    t = data["t"]

    z_flux = data["depth_flux"]
    z_profile = data["depth_profile"]

    z_max = maximum([maximum(z_flux), maximum(z_profile)])
    z_min = minimum([minimum(z_flux), minimum(z_profile)])

    anim = @animate for i in 1:length(t)
        @info "Animating $flux_type/$profile_type frame $i/$(length(t))"
        l = @layout [a b]
        fig₁ = plot(truth_flux[:,i], z_flux, xlim=(flux_min, flux_max), ylim=(z_min, z_max), label="Truth")
        plot!(fig₁, test_flux[:,i], z_flux, label = "NN")
        ylabel!(fig₁, "z /m")
        if dimensionless
            xlabel!(fig₁, flux_type)
        elseif flux_type == "wT"
            xlabel!(fig₁, "$flux_type /m s⁻¹ °C")
        else
            xlabel!(fig₁, "$flux_type /m² s⁻²")
        end

        fig₂ = plot(truth_profile[:,i], z_profile, xlim=(profile_min, profile_max), ylim=(z_min, z_max), label="Truth", legend=:topleft)
        plot!(fig₂, test_profile[:,i], z_profile, label="NN")
        ylabel!(fig₂, "z /m")
        if dimensionless
            xlabel!(fig₂, profile_type)
        elseif profile_type == "T"
            xlabel!(fig₂, "T /°C")
        else
            xlabel!(fig₂, "$profile_type /m s⁻¹")
        end

        fig = plot(fig₁, fig₂, layout=l, title="$(round(t[i]/86400, digits=2)) days")
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profiles(data, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true)
    truth_u = data["truth_u"]
    truth_v = data["truth_v"]
    truth_T = data["truth_T"]

    test_u = data["test_u"]
    test_v = data["test_v"]
    test_T = data["test_T"]

    u_max = maximum([maximum(truth_u), maximum(test_u)])
    u_min = minimum([minimum(truth_u), minimum(test_u)])

    v_max = maximum([maximum(truth_v), maximum(test_v)])
    v_min = minimum([minimum(truth_v), minimum(test_v)])
    
    T_max = maximum([maximum(truth_T), maximum(test_T)])
    T_min = minimum([minimum(truth_T), minimum(test_T)])

    t = data["t"]

    z = data["depth_profile"]

    z_max = maximum(z)
    z_min = minimum(z)

    anim = @animate for i in 1:length(t)
        if i % 50 == 0
            @info "Animating frame $i/$(length(t))"
        end
        l = @layout [a b c]
        fig₁ = Plots.plot(truth_u[:,i], z, xlim=(u_min, u_max), ylim=(z_min, z_max), label="Truth", legend=:bottomright)
        Plots.plot!(fig₁, test_u[:,i], z, label = "NN")
        Plots.ylabel!(fig₁, "z /m")
        if dimensionless
            Plots.xlabel!(fig₁, "u")
        else
            Plots.xlabel!(fig₁, "u /m s⁻¹")
        end

        fig₂ = Plots.plot(truth_v[:,i], z, xlim=(v_min, v_max), ylim=(z_min, z_max), label="Truth", legend=:bottomleft)
        Plots.plot!(fig₂, test_v[:,i], z, label = "NN")
        Plots.ylabel!(fig₂, "z /m")
        if dimensionless
            Plots.xlabel!(fig₂, "v")
        else
            Plots.xlabel!(fig₂, "v /m s⁻¹")
        end

        fig₃ = Plots.plot(truth_T[:,i], z, xlim=(T_min, T_max), ylim=(z_min, z_max), label="Truth", legend=:bottomright)
        Plots.plot!(fig₃, test_T[:,i], z, label = "NN")
        Plots.ylabel!(fig₃, "z /m")
        if dimensionless
            Plots.xlabel!(fig₃, "T")
        else
            Plots.xlabel!(fig₃, "T /°C")
        end

        fig = Plots.plot(fig₁, fig₂, fig₃, layout=l, title="$(round(t[i]/86400, digits=2)) days")
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end

function animate_profiles_fluxes(data, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true, SIMULATION_NAME="")
    times = data["t"]

    frame = Node(1)

    truth_u = @lift data["truth_u"][:,$frame]
    truth_v = @lift data["truth_v"][:,$frame]
    truth_T = @lift data["truth_T"][:,$frame]

    test_u = @lift data["test_u"][:,$frame]
    test_v = @lift data["test_v"][:,$frame]
    test_T = @lift data["test_T"][:,$frame]

    truth_uw = @lift data["truth_uw"][:,$frame]
    truth_vw = @lift data["truth_vw"][:,$frame]
    truth_wT = @lift data["truth_wT"][:,$frame]

    test_uw = @lift data["test_uw"][:,$frame]
    test_vw = @lift data["test_vw"][:,$frame]
    test_wT = @lift data["test_wT"][:,$frame]


    truth_Ri = @lift clamp.(data["truth_Ri"][:,$frame], -1, 2)
    test_Ri = @lift clamp.(data["test_Ri"][:,$frame], -1, 2)

    u_max = maximum([maximum(data["truth_u"]), maximum(data["test_u"])])
    u_min = minimum([minimum(data["truth_u"]), minimum(data["test_u"])])

    v_max = maximum([maximum(data["truth_v"]), maximum(data["test_v"])])
    v_min = minimum([minimum(data["truth_v"]), minimum(data["test_v"])])

    T_max = maximum([maximum(data["truth_T"]), maximum(data["test_T"])])
    T_min = minimum([minimum(data["truth_T"]), minimum(data["test_T"])])

    uw_max = maximum([maximum(data["truth_uw"]), maximum(data["test_uw"])])
    uw_min = minimum([minimum(data["truth_uw"]), minimum(data["test_uw"])])

    vw_max = maximum([maximum(data["truth_vw"]), maximum(data["test_vw"])])
    vw_min = minimum([minimum(data["truth_vw"]), minimum(data["test_vw"])])
    
    wT_max = maximum([maximum(data["truth_wT"]), maximum(data["test_wT"])])
    wT_min = minimum([minimum(data["truth_wT"]), minimum(data["test_wT"])])

    plot_title = @lift "$SIMULATION_NAME: time = $(round(times[$frame]/86400, digits=2)) days, loss = $(round(data["loss"], sigdigits=3))"
    fig = Figure(resolution=(1920, 1080))
    colors=["navyblue", "hotpink2"]

    if dimensionless
        u_str = "u"
        v_str = "v"
        T_str = "T"
        uw_str = "uw"
        vw_str = "vw"
        wT_str = "wT"
    else
        u_str = "u / m s⁻¹"
        v_str = "v / m s⁻¹"
        T_str = "T / °C"
        uw_str = "uw / m² s⁻²"
        vw_str = "vw / m² s⁻²"
        wT_str = "wT / m s⁻¹ °C"
    end

    zc = data["depth_profile"]
    zf = data["depth_flux"]
    z_str = "z / m"

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, truth_u, zc, linewidth=3, color=colors[1]), lines!(ax_u, test_u, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, truth_v, zc, linewidth=3, color=colors[1]), lines!(ax_v, test_v, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, truth_T, zc, linewidth=3, color=colors[1]), lines!(ax_T, test_T, zc, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, truth_uw, zf, linewidth=3, color=colors[1]), lines!(ax_uw, test_uw, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, truth_vw, zf, linewidth=3, color=colors[1]), lines!(ax_vw, test_vw, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, truth_wT, zf, linewidth=3, color=colors[1]), lines!(ax_wT, test_wT, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[2, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, truth_Ri, zf, linewidth=3, color=colors[1]), lines!(ax_Ri, test_Ri, zf, linewidth=3, color=colors[2])]
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NDE Prediction"])
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    trim!(fig.layout)

    if gif
        record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            @info "Animating gif frame $n/$(length(times))..."
            frame[] = n
        end
    end

    if mp4
        record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            @info "Animating mp4 frame $n/$(length(times))..."
            frame[] = n
        end
    end
end

function animate_profiles_fluxes_comparison(data, FILE_PATH; dimensionless=true, fps=30, gif=false, mp4=true, SIMULATION_NAME="")
    times = data["t"] ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

    truth_u = @lift data["truth_u"][:,$frame]
    truth_v = @lift data["truth_v"][:,$frame]
    truth_T = @lift data["truth_T"][:,$frame]

    test_u = @lift data["test_u"][:,$frame]
    test_v = @lift data["test_v"][:,$frame]
    test_T = @lift data["test_T"][:,$frame]

    truth_uw = @lift data["truth_uw"][:,$frame]
    truth_vw = @lift data["truth_vw"][:,$frame]
    truth_wT = @lift data["truth_wT"][:,$frame]

    test_uw = @lift data["test_uw"][:,$frame]
    test_vw = @lift data["test_vw"][:,$frame]
    test_wT = @lift data["test_wT"][:,$frame]

    test_u_modified_pacanowski_philander = @lift data["test_u_modified_pacanowski_philander"][:,$frame]
    test_v_modified_pacanowski_philander = @lift data["test_v_modified_pacanowski_philander"][:,$frame]
    test_T_modified_pacanowski_philander = @lift data["test_T_modified_pacanowski_philander"][:,$frame]

    test_uw_modified_pacanowski_philander = @lift data["test_uw_modified_pacanowski_philander"][:,$frame]
    test_vw_modified_pacanowski_philander = @lift data["test_vw_modified_pacanowski_philander"][:,$frame]
    test_wT_modified_pacanowski_philander = @lift data["test_wT_modified_pacanowski_philander"][:,$frame]

    # test_u_NN_only = @lift data["test_u_NN_only"][:,$frame]
    # test_v_NN_only = @lift data["test_v_NN_only"][:,$frame]
    # test_T_NN_only = @lift data["test_T_NN_only"][:,$frame]

    test_uw_NN_only = @lift data["test_uw_NN_only"][:,$frame]
    test_vw_NN_only = @lift data["test_vw_NN_only"][:,$frame]
    test_wT_NN_only = @lift data["test_wT_NN_only"][:,$frame]

    truth_Ri = @lift clamp.(data["truth_Ri"][:,$frame], -1, 2)
    test_Ri = @lift clamp.(data["test_Ri"][:,$frame], -1, 2)
    test_Ri_modified_pacanowski_philander = @lift clamp.(data["test_Ri_modified_pacanowski_philander"][:,$frame], -1, 2)
    # test_Ri_NN_only = @lift clamp.(data["test_Ri_NN_only"][:,$frame], -1, 2)

    losses = data["losses"]
    losses_gradient = data["losses_gradient"]
    losses_modified_pacanowski_philander = data["losses_modified_pacanowski_philander"]
    losses_modified_pacanowski_philander_gradient = data["losses_modified_pacanowski_philander_gradient"]

    losses .= losses .+ (losses .== 0) .* eps(Float32)
    losses_gradient .= losses_gradient .+ (losses_gradient .== 0) .* eps(Float32)
    losses_modified_pacanowski_philander .= losses_modified_pacanowski_philander .+ (
                                                 losses_modified_pacanowski_philander .== 0) .* eps(Float32)
    losses_modified_pacanowski_philander_gradient .= losses_modified_pacanowski_philander_gradient .+ (
                                                          losses_modified_pacanowski_philander_gradient .== 0) .* eps(Float32)

    loss_point = @lift [losses[$frame]]

    loss_gradient_point = @lift [losses_gradient[$frame]]
    loss_modified_pacanowski_philander_point = @lift [losses_modified_pacanowski_philander[$frame]]
    loss_modified_pacanowski_philander_gradient_point = @lift [losses_modified_pacanowski_philander_gradient[$frame]]

    u_max = maximum([maximum(data["truth_u"]), maximum(data["test_u"]), maximum(data["test_u_modified_pacanowski_philander"])])
    u_min = minimum([minimum(data["truth_u"]), minimum(data["test_u"]), minimum(data["test_u_modified_pacanowski_philander"])])

    v_max = maximum([maximum(data["truth_v"]), maximum(data["test_v"]), maximum(data["test_v_modified_pacanowski_philander"])])
    v_min = minimum([minimum(data["truth_v"]), minimum(data["test_v"]), minimum(data["test_v_modified_pacanowski_philander"])])

    T_max = maximum([maximum(data["truth_T"]), maximum(data["test_T"]), maximum(data["test_T_modified_pacanowski_philander"])])
    T_min = minimum([minimum(data["truth_T"]), minimum(data["test_T"]), minimum(data["test_T_modified_pacanowski_philander"])])

    uw_max = maximum([maximum(data["truth_uw"]), maximum(data["test_uw"]), maximum(data["test_uw_modified_pacanowski_philander"]), maximum(data["test_uw_NN_only"])])
    uw_min = minimum([minimum(data["truth_uw"]), minimum(data["test_uw"]), minimum(data["test_uw_modified_pacanowski_philander"]), minimum(data["test_uw_NN_only"])])

    vw_max = maximum([maximum(data["truth_vw"]), maximum(data["test_vw"]), maximum(data["test_vw_modified_pacanowski_philander"]), maximum(data["test_vw_NN_only"])])
    vw_min = minimum([minimum(data["truth_vw"]), minimum(data["test_vw"]), minimum(data["test_vw_modified_pacanowski_philander"]), minimum(data["test_vw_NN_only"])])
    
    wT_max = maximum([maximum(data["truth_wT"]), maximum(data["test_wT"]), maximum(data["test_wT_modified_pacanowski_philander"]), maximum(data["test_wT_NN_only"])])
    wT_min = minimum([minimum(data["truth_wT"]), minimum(data["test_wT"]), minimum(data["test_wT_modified_pacanowski_philander"]), minimum(data["test_wT_NN_only"])])

    losses_max = maximum([maximum(data["losses"]), maximum(data["losses_gradient"]), maximum(data["losses_modified_pacanowski_philander"]), maximum(data["losses_modified_pacanowski_philander_gradient"])])
    losses_min = minimum([minimum(data["losses"][2:end]), minimum(data["losses_gradient"][2:end]), minimum(data["losses_modified_pacanowski_philander"][2:end]), minimum(data["losses_modified_pacanowski_philander_gradient"][2:end])])

    plot_title = @lift "$SIMULATION_NAME: Time = $(round(times[$frame], digits=2)) days"
    fig = Figure(resolution=(1920, 1080))
    color_palette = distinguishable_colors(9, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    colors = (truth=color_palette[1], 
              test=color_palette[2], 
              test_modified_pacanoswki_philander=color_palette[3],
              test_NN_only=color_palette[4])
    colors_losses = (loss=color_palette[5],
                     loss_modified_pacanowski_philander=color_palette[6],
                     loss_gradient=color_palette[7],
                     loss_gradient_modified_pacanowski_philander=color_palette[8],
                     point=color_palette[9])

    if dimensionless
        u_str = "u"
        v_str = "v"
        T_str = "T"
        uw_str = "uw"
        vw_str = "vw"
        wT_str = "wT"
    else
        u_str = "u / m s⁻¹"
        v_str = "v / m s⁻¹"
        T_str = "T / °C"
        uw_str = "uw / m² s⁻²"
        vw_str = "vw / m² s⁻²"
        wT_str = "wT / m s⁻¹ °C"
    end

    zc = data["depth_profile"]
    zf = data["depth_flux"]
    z_str = "z / m"

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, truth_u, zc, linewidth=3, color=colors.truth), 
                lines!(ax_u, test_u_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_u, test_u, zc, linewidth=3, color=colors.test)
                ]
                # lines!(ax_u, test_u_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, truth_v, zc, linewidth=3, color=colors.truth), 
                lines!(ax_v, test_v_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_v, test_v, zc, linewidth=3, color=colors.test),
                ]
                # lines!(ax_v, test_v_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, truth_T, zc, linewidth=3, color=colors.truth), 
                lines!(ax_T, test_T_modified_pacanowski_philander, zc, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_T, test_T, zc, linewidth=3, color=colors.test)
                ]
                # lines!(ax_T, test_T_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, truth_uw, zf, linewidth=3, color=colors.truth), 
                lines!(ax_uw, test_uw_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_uw, test_uw_NN_only, zf, linewidth=3, color=colors.test_NN_only),
                lines!(ax_uw, test_uw, zf, linewidth=3, color=colors.test), 
                ]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, truth_vw, zf, linewidth=3, color=colors.truth), 
                lines!(ax_vw, test_vw_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_vw, test_vw_NN_only, zf, linewidth=3, color=colors.test_NN_only),
                lines!(ax_vw, test_vw, zf, linewidth=3, color=colors.test)]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, truth_wT, zf, linewidth=3, color=colors.truth), 
                lines!(ax_wT, test_wT_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_wT, test_wT_NN_only, zf, linewidth=3, color=colors.test_NN_only),
                lines!(ax_wT, test_wT, zf, linewidth=3, color=colors.test)]
                CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[1, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, truth_Ri, zf, linewidth=3, color=colors.truth), 
                lines!(ax_Ri, test_Ri_modified_pacanowski_philander, zf, linewidth=3, color=colors.test_modified_pacanoswki_philander),
                lines!(ax_Ri, test_Ri, zf, linewidth=3, color=colors.test)]
                # lines!(ax_Ri, test_Ri_NN_only, zf, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    ax_losses = fig[2, 4] = Axis(fig, xlabel="Time / days", ylabel="Loss", yscale=CairoMakie.log10)
    losses_lines = [lines!(ax_losses, times, losses, linewidth=3, color=colors_losses.loss),
                    lines!(ax_losses, times, losses_modified_pacanowski_philander, linewidth=3, color=colors_losses.loss_modified_pacanowski_philander),
                    lines!(ax_losses, times, losses_gradient, linewidth=3, color=colors_losses.loss_gradient),
                    lines!(ax_losses, times, losses_modified_pacanowski_philander_gradient, linewidth=3, color=colors_losses.loss_gradient_modified_pacanowski_philander)]
    losses_point = [CairoMakie.scatter!(ax_losses, time_point, loss_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_gradient_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_modified_pacanowski_philander_point, color=colors_losses.point),
                    CairoMakie.scatter!(ax_losses, time_point, loss_modified_pacanowski_philander_gradient_point, color=colors_losses.point)]                
    
    CairoMakie.xlims!(ax_losses, times[1], times[end])
    CairoMakie.ylims!(ax_losses, losses_min, losses_max)

    legend = fig[1, 5] = Legend(fig, uw_lines, ["Oceananigans.jl LES", 
                                                "Modified Pac-Phil Only", 
                                                "NN Only",
                                                "NN + Modified Pac-Phil"])
    legend = fig[2, 5] = Legend(fig, losses_lines, ["Profile Loss, NN + Modified Pac-Phil", 
                                                    "Profile Loss, Modified Pac-Phil Only", 
                                                    "Gradient Loss, NN + Modified Pac-Phil", 
                                                    "Gradient Loss, Modified Pac-Phil Only"])
    # legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NN + Modified Pac-Phil", "Modified Pac-Phil Only"])
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    trim!(fig.layout)

    if gif
        record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            @info "Animating gif frame $n/$(length(times))..."
            frame[] = n
        end
    end

    if mp4
        record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            @info "Animating mp4 frame $n/$(length(times))..."
            frame[] = n
        end
    end
end

function animate_training_data_profiles_fluxes(train_files, FILE_PATH; fps=30, gif=false, mp4=true)
    all_data = [WindMixing.data(train_file, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true) for train_file in train_files]

    times = all_data[1].t ./ 86400

    frame = Node(1)

    time_point = @lift [times[$frame]]

    u_data = [data.u.coarse for data in all_data]
    v_data = [data.v.coarse for data in all_data]
    T_data = [data.T.coarse for data in all_data]

    us = [@lift u[:,$frame] for u in u_data]
    vs = [@lift v[:,$frame] for v in v_data]
    Ts = [@lift T[:,$frame] for T in T_data]

    uws = [@lift data.uw.coarse[:,$frame] for data in all_data]
    vws = [@lift data.vw.coarse[:,$frame] for data in all_data]
    wTs = [@lift data.wT.coarse[:,$frame] for data in all_data]

    u_max = maximum(maximum(data.u.coarse) for data in all_data)
    u_min = minimum(minimum(data.u.coarse) for data in all_data)

    v_max = maximum(maximum(data.v.coarse) for data in all_data)
    v_min = minimum(minimum(data.v.coarse) for data in all_data)

    T_max = maximum(maximum(data.T.coarse) for data in all_data)
    T_min = minimum(minimum(data.T.coarse) for data in all_data)

    uw_max = maximum(maximum(data.uw.coarse) for data in all_data)
    uw_min = minimum(minimum(data.uw.coarse) for data in all_data)

    vw_max = maximum(maximum(data.vw.coarse) for data in all_data)
    vw_min = minimum(minimum(data.vw.coarse) for data in all_data)
    
    wT_max = maximum(maximum(data.wT.coarse) for data in all_data)
    wT_min = minimum(minimum(data.wT.coarse) for data in all_data)

    Nz = all_data[1].grid_points - 1
    zc = all_data[1].u.z
    zf = all_data[1].uw.z

    D_face = Float32.(Dᶠ(Nz, zc[2] - zc[1]))

    @inline function ∂_∂z(profile)
        output = zeros(typeof(profile[1]), size(profile, 1) + 1, size(profile,2))
        for i in 1:size(profile,2)
            profile_col = @view profile[:,i]
            output_col = @view output[:,i]
            output_col .= D_face * profile_col
        end
        return output
    end

    ∂u∂zs = [∂_∂z(u) for u in u_data]
    ∂v∂zs = [∂_∂z(v) for v in v_data]
    ∂T∂zs = [∂_∂z(T) for T in T_data]

    @inline function local_richardson(∂u∂z, ∂v∂z, ∂T∂z, g, α)
        # ϵ = eps(typeof(∂u∂z))
        ϵ = 0
        ∂u∂z += ϵ
        ∂v∂z += ϵ
        ∂T∂z += ϵ
        Bz = g * α * ∂T∂z
        S² = ∂u∂z ^2 + ∂v∂z ^2
        return clamp.(Bz / S², -1, 2)
    end

    α = 1.67f-4
    g = 9.81f0

    Ris_data = [local_richardson.(∂u∂zs[i], ∂v∂zs[i], ∂T∂zs[i], g, α) for i in 1:length(∂u∂zs)]

    Ris = [@lift Ri[:,$frame] for Ri in Ris_data]

    plot_title = @lift "LES Simulations: Time = $(round(times[$frame], digits=2)) days"
    fig = Figure(resolution=(1920, 1080))
    color_palette = distinguishable_colors(length(all_data), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    u_str = "u / m s⁻¹"
    v_str = "v / m s⁻¹"
    T_str = "T / °C"
    uw_str = "uw / m² s⁻²"
    vw_str = "vw / m² s⁻²"
    wT_str = "wT / m s⁻¹ °C"

    z_str = "z / m"

    uw_tops = [data.uw.coarse[end,1] for data in all_data]
    wT_tops = [data.wT.coarse[end,1] for data in all_data]

    BC_strs = [@sprintf "Momentum Flux = %.1e, Buoyancy Flux = %.1e" uw_tops[i] wT_tops[i] for i in 1:length(uw_tops)]

    ax_u = fig[1, 1] = Axis(fig, xlabel=u_str, ylabel=z_str)
    u_lines = [lines!(ax_u, us[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(us)]
                # lines!(ax_u, test_u_NN_only, zc, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_u, u_min, u_max)
    CairoMakie.ylims!(ax_u, minimum(zc), 0)

    ax_v = fig[1, 2] = Axis(fig, xlabel=v_str, ylabel=z_str)
    v_lines = [lines!(ax_v, vs[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(vs)]
    CairoMakie.xlims!(ax_v, v_min, v_max)
    CairoMakie.ylims!(ax_v, minimum(zc), 0)

    ax_T = fig[1, 3] = Axis(fig, xlabel=T_str, ylabel=z_str)
    T_lines = [lines!(ax_T, Ts[i], zc, linewidth=3, color=color_palette[i]) for i in 1:length(Ts)]
    CairoMakie.xlims!(ax_T, T_min, T_max)
    CairoMakie.ylims!(ax_T, minimum(zc), 0)

    ax_uw = fig[2, 1] = Axis(fig, xlabel=uw_str, ylabel=z_str)
    uw_lines = [lines!(ax_uw, uws[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(uws)]
    CairoMakie.xlims!(ax_uw, uw_min, uw_max)
    CairoMakie.ylims!(ax_uw, minimum(zf), 0)

    ax_vw = fig[2, 2] = Axis(fig, xlabel=vw_str, ylabel=z_str)
    vw_lines = [lines!(ax_vw, vws[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(vws)]
    CairoMakie.xlims!(ax_vw, vw_min, vw_max)
    CairoMakie.ylims!(ax_vw, minimum(zf), 0)

    ax_wT = fig[2, 3] = Axis(fig, xlabel=wT_str, ylabel=z_str)
    wT_lines = [lines!(ax_wT, wTs[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(wTs)]
    CairoMakie.xlims!(ax_wT, wT_min, wT_max)
    CairoMakie.ylims!(ax_wT, minimum(zf), 0)

    ax_Ri = fig[2, 4] = Axis(fig, xlabel="Ri", ylabel=z_str)
    Ri_lines = [lines!(ax_Ri, Ris[i], zf, linewidth=3, color=color_palette[i]) for i in 1:length(Ris)]
                # lines!(ax_Ri, test_Ri_NN_only, zf, linewidth=3, color=colors[4])]
    CairoMakie.xlims!(ax_Ri, -1, 2)
    CairoMakie.ylims!(ax_Ri, minimum(zf), 0)

    legend = fig[1, 4] = Legend(fig, uw_lines, BC_strs)

    # legend = fig[1, 4] = Legend(fig, u_lines, ["Oceananigans.jl LES", "NN + Modified Pac-Phil", "Modified Pac-Phil Only"])
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    trim!(fig.layout)

    if gif
        record(fig, "$FILE_PATH.gif", 1:length(times), framerate=fps) do n
            @info "Animating gif frame $n/$(length(times))..."
            frame[] = n
        end
    end

    if mp4
        record(fig, "$FILE_PATH.mp4", 1:length(times), framerate=fps) do n
            @info "Animating mp4 frame $n/$(length(times))..."
            frame[] = n
        end
    end
end