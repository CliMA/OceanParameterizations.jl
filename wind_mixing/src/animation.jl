function animate_NN(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str, PATH=joinpath(pwd(), "Output"))
    anim = @animate for n in 1:size(xs[1], 2)
        x_max = maximum(maximum(x) for x in xs)
        x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1], 2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n] / 86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    # gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
    mp4(anim, joinpath(PATH, "$(filename).mp4"), fps=30)
end

function prepare_parameters(𝒟train, uw_NN, vw_NN, wT_NN, f=1f-4, Nz=32)
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

function NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange)
    f, H, τ, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters(𝒟train, uw_NN, vw_NN, wT_NN)

    uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = prepare_BCs(𝒟test, uw_scaling, vw_scaling, wT_scaling)

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

    t_test = Float32.(𝒟test.t[trange] ./ τ)
    tspan_test = (t_test[1], t_test[end])
    uvT₀ = [u_scaling(𝒟test.uvT_unscaled[1:Nz, 1]); v_scaling(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); T_scaling(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]
    BC = [uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom]
    prob = ODEProblem(NDE!, uvT₀, tspan_test)

    sol = Array(solve(prob, ROCK4(), p=[weights; BC], sensealg=InterpolatingAdjoint(), saveat=t_test))

    output = Dict()

    output["truth_uw"] = uw_scaling.(𝒟test.uw.coarse[:,trange])
    output["truth_vw"] = vw_scaling.(𝒟test.vw.coarse[:,trange])
    output["truth_wT"] = wT_scaling.(𝒟test.wT.coarse[:,trange])

    output["truth_u"] = u_scaling.(𝒟test.uvT_unscaled[1:Nz, trange])
    output["truth_v"] = v_scaling.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange])
    output["truth_T"] = T_scaling.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])

    test_uw = similar(output["truth_uw"])
    test_vw = similar(output["truth_vw"])
    test_wT = similar(output["truth_wT"])

    for i in 1:size(test_uw, 2)
        uw = @view test_uw[:,i]
        vw = @view test_vw[:,i]
        wT = @view test_wT[:,i]
        uw .= predict_NDE(uw_NN, @view(sol[:,i]), uw_top, uw_bottom)
        vw .= predict_NDE(vw_NN, @view(sol[:,i]), vw_top, vw_bottom)
        wT .= predict_NDE(wT_NN, @view(sol[:,i]), wT_top, wT_bottom)
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

    return output
end

function NDE_profile_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange)
    f, H, τ, Nz, u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling, μ_u, μ_v, σ_u, σ_v, σ_T, σ_uw, σ_vw, σ_wT, weights, re_uw, re_vw, re_wT, D_cell, D_face, size_uw_NN, size_vw_NN, size_wT_NN, uw_range, vw_range, wT_range = prepare_parameters(𝒟train, uw_NN, vw_NN, wT_NN)
    κ = 10f0

    uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom = prepare_BCs(𝒟test, uw_scaling, vw_scaling, wT_scaling)

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

    t_test = Float32.(𝒟test.t[trange] ./ τ)
    tspan_test = (t_test[1], t_test[end])
    uvT₀ = [u_scaling(𝒟test.uvT_unscaled[1:Nz, 1]); v_scaling(𝒟test.uvT_unscaled[Nz + 1:2Nz, 1]); T_scaling(𝒟test.uvT_unscaled[2Nz + 1:3Nz, 1])]
    BC = [uw_top, uw_bottom, vw_top, vw_bottom, wT_top, wT_bottom]
    prob = ODEProblem(NDE!, uvT₀, tspan_test)

    sol = Array(solve(prob, ROCK4(), p=[weights; BC], sensealg=InterpolatingAdjoint(), saveat=t_test))

    output = Dict()

    output["truth_uw"] = uw_scaling.(𝒟test.uw.coarse[:,trange])
    output["truth_vw"] = vw_scaling.(𝒟test.vw.coarse[:,trange])
    output["truth_wT"] = wT_scaling.(𝒟test.wT.coarse[:,trange])

    output["truth_u"] = u_scaling.(𝒟test.uvT_unscaled[1:Nz, trange])
    output["truth_v"] = v_scaling.(𝒟test.uvT_unscaled[Nz + 1:2Nz, trange])
    output["truth_T"] = T_scaling.(𝒟test.uvT_unscaled[2Nz + 1:3Nz, trange])

    test_uw = similar(output["truth_uw"])
    test_vw = similar(output["truth_vw"])
    test_wT = similar(output["truth_wT"])

    for i in 1:size(test_uw, 2)
        uw = @view test_uw[:,i]
        vw = @view test_vw[:,i]
        wT = @view test_wT[:,i]
        uw .= predict_NDE(uw_NN, @view(sol[:,i]), uw_top, uw_bottom)
        vw .= predict_NDE(vw_NN, @view(sol[:,i]), vw_top, vw_bottom)
        wT .= predict_NDE(wT_NN, @view(sol[:,i]), wT_top, wT_bottom)
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
        else
            xlabel!(fig₁, "$flux_type /m² s⁻²")
        end

        fig₂ = plot(truth_profile[:,i], z_profile, xlim=(profile_min, profile_max), ylim=(z_min, z_max), label="Truth")
        plot!(fig₂, test_profile[:,i], z_profile, label="NN")
        ylabel!(fig₂, "z /m")
        if dimensionless
            xlabel!(fig₂, profile_type)
        elseif profile_type == "T"
            xlabel!(fig₂, "T /K")
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