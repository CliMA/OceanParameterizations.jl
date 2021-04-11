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

function prepare_BCs(𝒟, uw_scaling, vw_scaling, wT_scaling)
    uw_top = uw_scaling(𝒟.uw.coarse[1,1])
    uw_bottom = uw_scaling(𝒟.uw.coarse[end,1])
    vw_top = vw_scaling(𝒟.vw.coarse[1,1])
    vw_bottom = vw_scaling(𝒟.vw.coarse[end,1])
    wT_top = wT_scaling(𝒟.wT.coarse[1,1])
    wT_bottom = wT_scaling(𝒟.wT.coarse[end,1])
    return uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom
end

function NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange; 
                    unscale=false, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, α=1.67f-4, g=9.81f0, 
                    modified_pacanowski_philander=false, convective_adjustment=false,
                    smooth_NN=false, smooth_Ri=false)
    f, H, τ, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters_NDE_animation(𝒟train, uw_NN, vw_NN, wT_NN)

    uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = prepare_BCs(𝒟test, uw_scaling, vw_scaling, wT_scaling)

    @assert !modified_pacanowski_philander || !convective_adjustment

    if smooth_NN
        filter_interior = WindMixing.smoothing_filter(Nz-1, 3) 
     end
 
    if smooth_Ri
        filter_face = WindMixing.smoothing_filter(Nz+1, 3) 
    end

    tanh_step(x) = (1 - tanh(x)) / 2

    ϵ = 1f-7

    function predict_NDE(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:3Nz]

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

        uw = [uw_top; uw_interior; uw_bottom]
        vw = [vw_top; vw_interior; vw_bottom]
        wT = [wT_top; wT_interior; wT_bottom]

        if modified_pacanowski_philander
            ∂u∂z = D_face * u
            ∂v∂z = D_face * v
            ∂T∂z = D_face * T
            Ri = local_richardson.(∂u∂z.+ 1f-7, ∂v∂z.+ 1f-7, ∂T∂z.+ 1f-7, σ_u, σ_v, σ_T, H, g, α)

            if smooth_Ri
                Ri = filter_face * Ri
            end

            ν = ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)
            ∂z_ν∂u∂z = D_cell * (ν .* ∂u∂z)
            ∂z_ν∂v∂z = D_cell * (ν .* ∂v∂z)
            ∂z_ν∂T∂z = D_cell * (ν .* ∂T∂z ./ Pr)
            ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v) .+ τ / H ^ 2 .* ∂z_ν∂u∂z
            ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u) .+ τ / H ^ 2 .* ∂z_ν∂v∂z
            ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT .+ τ / H ^ 2 .* ∂z_ν∂T∂z
        elseif convective_adjustment
            ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
            ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
            ∂T∂z = D_face * T
            ∂z_∂T∂z = D_cell * min.(0f0, ∂T∂z)
            ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT .+ τ / H ^2 * κ .* ∂z_∂T∂z
        else
            ∂u∂t = -τ / H * σ_uw / σ_u .* D_cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
            ∂v∂t = -τ / H * σ_vw / σ_v .* D_cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
            ∂T∂t = -τ / H * σ_wT / σ_T .* D_cell * wT
        end

        return [∂u∂t; ∂v∂t; ∂T∂t]
    end

    function predict_flux(uw_NN, vw_NN, wT_NN, x, uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        u = @view x[1:Nz]
        v = @view x[Nz + 1:2Nz]
        T = @view x[2Nz + 1:3Nz]

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

        uw = [uw_top; uw_interior; uw_bottom]
        vw = [vw_top; vw_interior; vw_bottom]
        wT = [wT_top; wT_interior; wT_bottom]

        if modified_pacanowski_philander
            ∂u∂z = D_face * u
            ∂v∂z = D_face * v
            ∂T∂z = D_face * T
            Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, σ_u, σ_v, σ_T, H, g, α)
            
            if smooth_Ri
                Ri = filter_face * Ri
            end

            ν = ν₀ .+ ν₋ .* tanh_step.((Ri .- Riᶜ) ./ ΔRi)
            uw .- ν ./ H .* σ_u ./ σ_uw .* ∂u∂z
            vw .- ν ./ H .* σ_v ./ σ_vw .* ∂v∂z
            wT .- ν ./ H .* σ_T ./ σ_wT .* ∂T∂z ./ Pr
        elseif convective_adjustment
            ∂T∂z = D_face * T
            wT .- σ_T / (σ_wT * H) .* min.(0f0, ∂T∂z) .* κ
        end

        return uw, vw, wT
    end

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

    t_test = Float32.(𝒟test.t[trange] ./ τ)
    tspan_test = (t_test[1], t_test[end])
    uvT₀ = [u_scaling(𝒟test.uvT_unscaled[1:Nz, 1]); v_scaling(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); T_scaling(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]
    BC = [uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom]
    prob = ODEProblem(NDE, uvT₀, tspan_test)

    sol = Array(solve(prob, ROCK4(), p=[weights; BC], saveat=t_test))

    output = Dict()

    if unscale
        output["truth_uw"] = 𝒟test.uw.coarse[:,trange]
        output["truth_vw"] = 𝒟test.vw.coarse[:,trange]
        output["truth_wT"] = 𝒟test.wT.coarse[:,trange]

        output["truth_u"] = 𝒟test.uvT_unscaled[1:Nz, trange]
        output["truth_v"] = 𝒟test.uvT_unscaled[Nz + 1:2Nz, trange]
        output["truth_T"] = 𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange]

        test_uw = similar(output["truth_uw"])
        test_vw = similar(output["truth_vw"])
        test_wT = similar(output["truth_wT"])

        Threads.@threads for i in 1:size(test_uw, 2)
            test_uw[:,i], test_vw[:,i], test_wT[:,i] = predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        end

        output["test_uw"] = inv(uw_scaling).(test_uw)
        output["test_vw"] = inv(vw_scaling).(test_vw)
        output["test_wT"] = inv(wT_scaling).(test_wT)

        output["test_u"] = inv(u_scaling).(sol[1:Nz,:])
        output["test_v"] = inv(v_scaling).(sol[Nz + 1:2Nz, :])
        output["test_T"] = inv(T_scaling).(sol[2Nz + 1: 3Nz, :])

        truth_Ri = similar(test_uw)

        for i in 1:size(truth_Ri, 2)
            truth_Ri[:,i] .= local_richardson.(D_face * 𝒟train.u.scaled[:,i], D_face * 𝒟train.v.scaled[:,i], D_face * 𝒟train.T.scaled[:,i], σ_u, σ_v, σ_T, H, g, α)
        end

        test_Ri = similar(truth_Ri)

        for i in 1:size(test_Ri,2)
            test_Ri[:,i] .= local_richardson.(D_face * sol[1:Nz,i], D_face * sol[Nz + 1:2Nz, i], D_face * sol[2Nz + 1: 3Nz, i], σ_u, σ_v, σ_T, H, g, α)
        end

        output["truth_Ri"] = truth_Ri
        output["test_Ri"] = test_Ri

        output["depth_profile"] = 𝒟test.u.z
        output["depth_flux"] = 𝒟test.uw.z
        output["t"] = 𝒟test.t[trange]
    else
        output["truth_uw"] = uw_scaling.(𝒟test.uw.coarse[:,trange])
        output["truth_vw"] = vw_scaling.(𝒟test.vw.coarse[:,trange])
        output["truth_wT"] = wT_scaling.(𝒟test.wT.coarse[:,trange])

        output["truth_u"] = u_scaling.(𝒟test.uvT_unscaled[1:Nz, trange])
        output["truth_v"] = v_scaling.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange])
        output["truth_T"] = T_scaling.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])

        test_uw = similar(output["truth_uw"])
        test_vw = similar(output["truth_vw"])
        test_wT = similar(output["truth_wT"])

        Threads.@threads for i in 1:size(test_uw, 2)
            test_uw[:,i], test_vw[:,i], test_wT[:,i] = predict_flux(uw_NN, vw_NN, wT_NN, @view(sol[:,i]), uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom)
        end

        output["test_uw"] = test_uw
        output["test_vw"] = test_vw
        output["test_wT"] = test_wT

        output["test_u"] = sol[1:Nz,:]
        output["test_v"] = sol[Nz + 1:2Nz, :]
        output["test_T"] = sol[2Nz + 1: 3Nz, :]
        output["depth_profile"] = 𝒟test.u.z
        output["depth_flux"] = 𝒟test.uw.z
        output["t"] = 𝒟test.t[trange]
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

    plot_title = @lift "$SIMULATION_NAME: time = $(round(times[$frame]/86400, digits=2)) days"
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


function animate_local_richardson_profile(uvT, 𝒟, FILE_PATH; α=1.67f-4, g=9.81f0, fps=30, gif=false, mp4=true, unscale=false)
    H = Float32(abs(𝒟.uw.z[end] - 𝒟.uw.z[1]))
    σ_u = Float32(𝒟.scalings["u"].σ)
    σ_v = Float32(𝒟.scalings["v"].σ)
    σ_T = Float32(𝒟.scalings["T"].σ)
    Ris = local_richardson(uvT, 𝒟, unscale=unscale)
    t = 𝒟.t
    z = 𝒟.uw.z

    z_max = maximum(z)
    z_min = minimum(z)

    Ri_max = maximum(Ris)
    Ri_min = minimum(Ris)

    @info "$Ri_min, $Ri_max"
    
    anim = @animate for i in 1:length(t)
        @info "Animating local Richardson number frame $i/$(length(t))"
        fig = plot(Ris[:,i], z, xlim=(Ri_min, Ri_max), ylim=(z_min, z_max), label=nothing, title="$(round(t[i]/86400, digits=2)) days", scale=:log10)
        ylabel!(fig, "z /m")
        xlabel!(fig, "Local Richardson Number")
    end

    if gif
        Plots.gif(anim, "$FILE_PATH.gif", fps=fps)
    end

    if mp4
        Plots.mp4(anim, "$FILE_PATH.mp4", fps=fps)
    end
end