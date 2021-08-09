
function prepare_parameters_NDE_training(uw_NN, vw_NN, wT_NN, Nz)
    uw_weights, re_uw = Flux.destructure(uw_NN)
    vw_weights, re_vw = Flux.destructure(vw_NN)
    wT_weights, re_wT = Flux.destructure(wT_NN)

    size_uw_NN = length(uw_weights)
    size_vw_NN = length(vw_weights)
    size_wT_NN = length(wT_weights)

    uw_range = 1:size_uw_NN
    vw_range = size_uw_NN + 1:size_uw_NN + size_vw_NN
    wT_range = size_uw_NN + size_vw_NN + 1:size_uw_NN + size_vw_NN + size_wT_NN

    derivatives = (cell=Float32.(Dᶜ(Nz, 1 / Nz)), face=Float32.(Dᶠ(Nz, 1 / Nz)))
    NN_constructions = (uw=re_uw, vw=re_vw, wT=re_wT)
    weights = Float32[uw_weights; vw_weights; wT_weights]

    NN_ranges = (uw=uw_range, vw=vw_range, wT=wT_range)

    return derivatives, NN_constructions, weights, NN_ranges
end

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
        if conditions.convective_adjustment
            constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, ν₀=ν₀, ν₋=ν₋, Riᶜ=Riᶜ, ΔRi=ΔRi, Pr=Pr, κ=κ)
        else
            constants = (H=H, τ=τ, f=f, Nz=Nz, g=g, α=α, ν₀=ν₀, ν₋=ν₋, Riᶜ=Riᶜ, ΔRi=ΔRi, Pr=Pr)
        end
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

function NDE(x, p, t, dataset, NN_ranges, NN_constructions, conditions, derivatives, diffusivity_scheme)
    uw_range, vw_range, wT_range = NN_ranges.uw, NN_ranges.vw, NN_ranges.wT
    uw_weights, vw_weights, wT_weights = p[uw_range], p[vw_range], p[wT_range]

    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom, wT_top = p[wT_range[end] + 1:end]
    BCs = (uw=(top=uw_top, bottom=uw_bottom), vw=(top=vw_top, bottom=vw_bottom), wT=(top=wT_top, bottom=wT_bottom))

    re_uw, re_vw, re_wT = NN_constructions.uw, NN_constructions.vw, NN_constructions.wT
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    return predict_NDE(uw_NN, vw_NN, wT_NN, x, dataset, BCs, conditions, derivatives, diffusivity_scheme)
end

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

function NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters, wT_top_function)
    uw_range, vw_range, wT_range = NN_ranges.uw, NN_ranges.vw, NN_ranges.wT
    uw_weights, vw_weights, wT_weights = p[uw_range], p[vw_range], p[wT_range]

    uw_bottom, uw_top, vw_bottom, vw_top, wT_bottom = p[wT_range[end] + 1:end-1]
    wT_top = scalings.wT(wT_top_function(t * constants.τ))
    
    BCs = (uw=(top=uw_top, bottom=uw_bottom), vw=(top=vw_top, bottom=vw_bottom), wT=(top=wT_top, bottom=wT_bottom))
    re_uw, re_vw, re_wT = NN_constructions.uw, NN_constructions.vw, NN_constructions.wT
    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)
    return predict_NDE(uw_NN, vw_NN, wT_NN, x, BCs, conditions, scalings, constants, derivatives, filters)
end

abstract type AbstractBaseDiffusivity end
struct RiBasedDiffusivity{T} <: AbstractBaseDiffusivity
    ν₀::T
    ν₋::T
    ΔRi::T
    Riᶜ::T
    Pr::T
end

function calculate_diffusive_flux(scheme::RiBasedDiffusivity, x, dataset, derivatives)
    Nz = dataset["u"].grid.Nz
    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]
    T = @view x[2Nz + 1:3Nz]

    H = dataset["u"].grid.Lz
    g = dataset.metadata["gravitational_acceleration"]
    α = dataset.metadata["thermal_expansion_coefficient"]

    scalings = dataset.metadata["scalings"]

    σ_u, σ_v, σ_T = scalings.u.σ, scalings.v.σ, scalings.T.σ
    σ_uw, σ_vw, σ_wT = scalings.uw.σ, scalings.vw.σ, scalings.wT.σ

    D_face = derivatives.face

    ∂u∂z = D_face * u
    ∂v∂z = D_face * v
    ∂T∂z = D_face * T

    ϵ = 1f-7
    Ri = local_richardson.(∂u∂z .+ ϵ, ∂v∂z .+ ϵ, ∂T∂z .+ ϵ, H, g, α, σ_u, σ_v, σ_T)

    ν_velocities = scheme.ν₀ .+ scheme.ν₋ .* tanh_step.((Ri .- scheme.Riᶜ) ./ scheme.ΔRi)
    ν_T = ν_velocities ./ scheme.Pr

    ν∂u∂z = σ_u / σ_uw / H .* ν_velocities .* ∂u∂z
    ν∂v∂z = σ_v / σ_vw / H .* ν_velocities .* ∂v∂z
    ν∂T∂z = σ_T / σ_wT / H .* ν_T .* ∂T∂z

    return ν∂u∂z, ν∂v∂z, ν∂T∂z
end

struct ConvectiveAdjustment{T} <: AbstractBaseDiffusivity
    κ::T
end

function calculate_diffusive_flux(scheme::ConvectiveAdjustment, x, dataset, derivatives)
    Nz = dataset["u"].grid.Nz

    T = @view x[2Nz + 1:3Nz]

    H = dataset["u"].grid.Lz

    ∂T∂z = derivatives.face * T

    scalings = dataset.metadata["scalings"]

    ν∂u∂z = zeros(Float32, Nz+1)
    ν∂v∂z = zeros(Float32, Nz+1)
    ν∂T∂z = scalings.T.σ / scalings.wT.σ / H .* scheme.κ .* min.(0f0, ∂T∂z)

    return ν∂u∂z, ν∂v∂z, ν∂T∂z
end

struct NoDiffusivity <: AbstractBaseDiffusivity
end

function calculate_diffusive_flux(scheme::NoDiffusivity, x, dataset, derivatives)
    Nz = dataset["u"].grid.Nz

    return zeros(Float32, Nz+1), zeros(Float32, Nz+1), zeros(Float32, Nz+1)
end

(scheme::AbstractBaseDiffusivity)(x, dataset, derivatives) = calculate_diffusive_flux(scheme, x, dataset, derivatives)

function predict_flux(uw_NN, vw_NN, wT_NN, x, dataset, BCs, conditions, derivatives, diffusivity_scheme)
    H = dataset["u"].grid.Lz

    scalings = dataset.metadata["scalings"]
    
    uw_scaling, vw_scaling, wT_scaling = scalings.uw, scalings.vw, scalings.wT
    σ_uw, σ_vw, σ_wT = uw_scaling.σ, vw_scaling.σ, wT_scaling.σ
    σ_u, σ_v, σ_T = scalings.u.σ, scalings.v.σ, scalings.T.σ
    
    ν∂u∂z, ν∂v∂z, ν∂T∂z = diffusivity_scheme(x, dataset, derivatives)

    uw_interior = uw_NN(x)
    vw_interior = vw_NN(x)
    wT_interior = wT_NN(x)
    
    if conditions.zero_weights
        uw = [0f0; uw_interior; 0f0]
        vw = [0f0; vw_interior; 0f0]
        wT = [0f0; wT_interior; 0f0]

        ν∂u∂z = [-(BCs.uw.bottom - scalings.uw(0f0)); ν∂u∂z[2:end-1]; -(BCs.uw.top - scalings.uw(0f0))]
        ν∂v∂z = [-(BCs.vw.bottom - scalings.vw(0f0)); ν∂v∂z[2:end-1]; -(BCs.vw.top - scalings.vw(0f0))]
        ν∂T∂z = [-(BCs.wT.bottom - scalings.wT(0f0)); ν∂T∂z[2:end-1]; -(BCs.wT.top - scalings.wT(0f0))]
    else
        uw = [BCs.uw.bottom; uw_interior; BCs.uw.top]
        vw = [BCs.vw.bottom; vw_interior; BCs.vw.top]
        wT = [BCs.wT.bottom; wT_interior; BCs.wT.top]
    end

    return uw .- ν∂u∂z, vw .- ν∂v∂z, wT .- ν∂T∂z
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

function predict_NDE(uw_NN, vw_NN, wT_NN, x, dataset, BCs, conditions, derivatives, diffusivity_scheme)
    Nz = Int64(dataset["u"].grid.Nz)
    H = dataset["u"].grid.Lz
    τ = dataset.metadata["τ"]
    f = dataset.metadata["coriolis_parameter"]

    @assert H > 0 && H isa Float32

    scalings = dataset.metadata["scalings"]

    σ_uw, σ_vw, σ_wT = scalings.uw.σ, scalings.vw.σ, scalings.wT.σ
    μ_u, μ_v, σ_u, σ_v, σ_T = scalings.u.μ, scalings.v.μ, scalings.u.σ, scalings.v.σ, scalings.T.σ

    u = @view x[1:Nz]
    v = @view x[Nz + 1:2Nz]

    uw, vw, wT = predict_flux(uw_NN, vw_NN, wT_NN, x, dataset, BCs, conditions, derivatives, diffusivity_scheme)
    
    ∂u∂t = -τ / H * σ_uw / σ_u .* derivatives.cell * uw .+ f * τ / σ_u .* (σ_v .* v .+ μ_v)
    ∂v∂t = -τ / H * σ_vw / σ_v .* derivatives.cell * vw .- f * τ / σ_v .* (σ_u .* u .+ μ_u)
    ∂T∂t = -τ / H * σ_wT / σ_T .* derivatives.cell * wT

    return [∂u∂t; ∂v∂t; ∂T∂t]
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

function prepare_training_data!(𝒟train, tsteps, n_simulations)
    u_trains = [interior(𝒟train.data[i]["u*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]
    v_trains = [interior(𝒟train.data[i]["v*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]
    T_trains = [interior(𝒟train.data[i]["T*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]

    uvT₀s = [
        [u_trains[i][:,1]; v_trains[i][:,1]; T_trains[i][:,1]] for i in 1:n_simulations
    ]

    ∂u∂zs = [interior(𝒟train.data[i]["∂u∂z*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]
    ∂v∂zs = [interior(𝒟train.data[i]["∂v∂z*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]
    ∂T∂zs = [interior(𝒟train.data[i]["∂T∂z*"])[1,1,:,tsteps[i][1]] for i in 1:n_simulations]

    t_trains = [Float32.(𝒟train.data[i]["u"].times[tsteps[i]]) for i in 1:n_simulations]
    @assert typeof(t_trains[1][1]) == Float32

    for i in 1:n_simulations
        𝒟train.data[i].metadata["τ"] = t_trains[i][end]
    end

    t_trains .= [times ./ times[end] for times in t_trains]
    tspan_trains = [(t[1], t[end]) for t in t_trains]

    profiles = (u=u_trains, v=v_trains, T=T_trains, ∂u∂z=∂u∂zs, ∂v∂z=∂v∂zs, ∂T∂z=∂T∂zs)
    times = (t=t_trains, tspan=tspan_trains)

    return profiles, uvT₀s, times
end

#TODO: add diurnal functionality
function train_NDE(uw_NN, vw_NN, wT_NN, train_files, tsteps, timestepper, optimizers, epochs, FILE_PATH; 
                    maxiters=500, diffusivity_scheme=RiBasedDiffusivity(1f-4, 1f-1, 1f-1, 0.25f0, 1f0), train_gradient=false,
                    zero_weights=false, gradient_scaling=5f-3, training_fractions=nothing, diurnal=false, 
                    Nz=32, scaling=ZeroMeanUnitVarianceScaling)

    train_parameters = Dict(
               "diffusivity_scheme" => diffusivity_scheme,
                   "train_gradient" => train_gradient,
                     "zero_weights" => zero_weights, 
                 "gradient_scaling" => gradient_scaling, 
               "training_fractions" => training_fractions,
                          "diurnal" => diurnal
    )

    @info "Loading training data"

    𝒟train = load_data(train_files, Nz_coarse=Nz, scaling=scaling)

    @info "Setting up constants"
    
    n_simulations = length(𝒟train.data)

    conditions = (train_gradient=train_gradient, zero_weights=zero_weights, diurnal=diurnal)
    
    derivatives, NN_constructions, weights, NN_ranges = prepare_parameters_NDE_training(uw_NN, vw_NN, wT_NN, Nz)

    @info "Setting up training data"

    !isa(tsteps, Array) && (tsteps = [tsteps for i in 1:n_simulations])

    training_profiles, uvT₀s, times = prepare_training_data!(𝒟train, tsteps, n_simulations)

    @info "Setting up equations and boundary conditions"

    BCs = [
        [
            interior(dataset["wu*"])[1,1,1,1], interior(dataset["wu*"])[1,1,end,1],
            interior(dataset["wv*"])[1,1,1,1], interior(dataset["wv*"])[1,1,end,1],
            interior(dataset["wT*"])[1,1,1,1], interior(dataset["wT*"])[1,1,end,1],
        ]
        for dataset in 𝒟train.data
    ]

    if diurnal
        # prob_NDEs = [
        #     ODEProblem(
        #         (x, p, t) -> NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters, wT_top_functions[i]), uvT₀s[i], tspan_train
        #     ) for i in 1:n_simulations
        # ]
        error("Diurnal Flux not Implemented yet")
    else
        prob_NDEs = [ODEProblem((x, p, t) -> NDE(x, p, t, 𝒟train.data[i], NN_ranges, NN_constructions, conditions, derivatives, diffusivity_scheme), 
                                    uvT₀s[i], times.tspan[i]) for i in 1:n_simulations]
    end

    function determine_loss_scalings()
        if training_fractions === nothing
            loss_scalings = (u=1, v=1, T=1, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
        else
            sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=times.t[i])) for i in 1:n_simulations]        
            u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)

            u_loss = [loss(training_profiles.u[i], u_sols[i]) for i in 1:n_simulations] |> mean
            v_loss = [loss(training_profiles.v[i], v_sols[i]) for i in 1:n_simulations] |> mean
            T_loss = [loss(training_profiles.T[i], T_sols[i]) for i in 1:n_simulations] |> mean
    
            if train_gradient
                u_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in u_sols]
                v_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in v_sols]
                T_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in T_sols]

                ∂u∂z_loss = [loss(training_profiles.∂u∂z[i], u_sols_gradients[i]) for i in 1:n_simulations] |> mean
                ∂v∂z_loss = [loss(training_profiles.∂v∂z[i], v_sols_gradients[i]) for i in 1:n_simulations] |> mean
                ∂T∂z_loss = [loss(training_profiles.∂T∂z[i], T_sols_gradients[i]) for i in 1:n_simulations] |> mean
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

    @info "Defining Loss Functions"

    function loss_NDE(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=times.t[i])) for i in 1:n_simulations]        
        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)

        u_loss = [loss(training_profiles.u[i], u_sols[i]) for i in 1:n_simulations] |> mean
        v_loss = [loss(training_profiles.v[i], v_sols[i]) for i in 1:n_simulations] |> mean
        T_loss = [loss(training_profiles.T[i], T_sols[i]) for i in 1:n_simulations] |> mean

        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=0, ∂v∂z=0, ∂T∂z=0)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses, loss_scalings
    end

    function loss_gradient_NDE(weights, BCs)
        sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=times.t[i])) for i in 1:n_simulations]
        # sols = [Array(solve(prob_NDEs[i], timestepper, p=[weights; BCs[i]], reltol=1f-3, saveat=t_trains[i])) for i in 1:n_simulations]

        u_sols, v_sols, T_sols = split_u.(sols, Nz), split_v.(sols, Nz), split_T.(sols, Nz)

        u_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in u_sols]
        v_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in v_sols]
        T_sols_gradients = [∂_∂z(sol, derivatives.face) for sol in T_sols]

        u_loss = [loss(training_profiles.u[i], u_sols[i]) for i in 1:n_simulations] |> mean
        v_loss = [loss(training_profiles.v[i], v_sols[i]) for i in 1:n_simulations] |> mean
        T_loss = [loss(training_profiles.T[i], T_sols[i]) for i in 1:n_simulations] |> mean

        ∂u∂z_loss = [loss(training_profiles.∂u∂z[i], u_sols_gradients[i]) for i in 1:n_simulations] |> mean
        ∂v∂z_loss = [loss(training_profiles.∂v∂z[i], v_sols_gradients[i]) for i in 1:n_simulations] |> mean
        ∂T∂z_loss = [loss(training_profiles.∂T∂z[i], T_sols_gradients[i]) for i in 1:n_simulations] |> mean
        
        losses = (u=u_loss, v=v_loss, T=T_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss)
        scaled_losses = apply_loss_scalings(losses, loss_scalings)

        return sum(scaled_losses), scaled_losses, loss_scalings, T_sols
    end

    @info "Setting up optimization problem"

    if train_gradient
        f_loss = OptimizationFunction(loss_gradient_NDE, GalacticOptim.AutoZygote())
    else
        f_loss = OptimizationFunction(loss_NDE, GalacticOptim.AutoZygote())
    end

    prob_loss = OptimizationProblem(f_loss, weights, BCs)

    @inline function rounded_percentage(num, den)
        return round(num / den * 100, sigdigits=3)
    end

    @info "Starting Training"
    for i in 1:length(optimizers), epoch in 1:epochs
        iter = 1
        opt = optimizers[i]
        function cb(args...)
            if iter <= maxiters
                total_loss = args[2]
                losses = args[3]
                loss_scalings = args[4]

                T_sols = args[5]

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

                @info "loss = $(total_loss): uvT$(profile_percent)% grad$(gradient_percent)% u$(u_percent)% v$(v_percent)% T$(T_percent)% ∂u∂z$(∂u∂z_percent)% ∂v∂z$(∂v∂z_percent)% ∂T∂z$(∂T∂z_percent)% opt$i/$(length(optimizers)) epoch$epoch/$epochs iter$iter/$maxiters"
                write_data_NDE_training(FILE_PATH, losses, loss_scalings,
                                    NN_constructions.uw(args[1][NN_ranges.uw]), 
                                    NN_constructions.vw(args[1][NN_ranges.vw]), 
                                    NN_constructions.wT(args[1][NN_ranges.wT]), 
                                    opt)
            end
            iter += 1
            false
        end
        res = solve(prob_loss, opt, cb=cb, maxiters=maxiters)
        weights .= res.minimizer
    end
    return NN_constructions.uw(weights[NN_ranges.uw]), NN_constructions.vw(weights[NN_ranges.vw]), NN_constructions.wT(weights[NN_ranges.wT])
end


function train_NDE(uw_NN, vw_NN, wT_NN, train_files, tsteps, timestepper, optimizers, epochs, FILE_PATH, stage; 
                    maxiters=500, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=2f-4, g=9.80665f0, 
                    modified_pacanowski_philander=false, convective_adjustment=false, smooth_profile=false, smooth_NN=false, smooth_Ri=false, train_gradient=false,
                    zero_weights=false, gradient_scaling=5f-3, training_fractions=nothing, diurnal=false)
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
               "training_fractions" => training_fractions,
                          "diurnal" => diurnal
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
                    train_gradient=train_gradient, zero_weights=zero_weights, diurnal=diurnal)
    
    constants, scalings, derivatives, NN_constructions, weights, NN_sizes, NN_ranges, filters = prepare_parameters_NDE_training(𝒟train, uw_NN, vw_NN, wT_NN, f, Nz, g, α, ν₀, ν₋, Riᶜ, ΔRi, Pr, κ, conditions)
    D_face = derivatives.face

    n_steps = Int(length(@view(𝒟train.t[:,1])) / n_simulations)

    if diurnal
        wT_top_functions = diurnal_fluxes(train_files, constants)
    end

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


    t_train = t_train ./ constants.τ
    tspan_train = (t_train[1], t_train[end])

    BCs = [[𝒟train.uw.scaled[1,n_steps * i + tsteps[1]],
        𝒟train.uw.scaled[end,n_steps * i + tsteps[1]],
        𝒟train.vw.scaled[1,n_steps * i + tsteps[1]],
        𝒟train.vw.scaled[end,n_steps * i + tsteps[1]],
        𝒟train.wT.scaled[1,n_steps * i + tsteps[1]],
        𝒟train.wT.scaled[end,n_steps * i + tsteps[1]]] for i in 0:n_simulations - 1]

    if diurnal
        prob_NDEs = [
            ODEProblem(
                (x, p, t) -> NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters, wT_top_functions[i]), uvT₀s[i], tspan_train
            ) for i in 1:n_simulations
        ]
    else
        prob_NDE(x, p, t) = NDE(x, p, t, NN_ranges, NN_constructions, conditions, scalings, constants, derivatives, filters)
        prob_NDEs = [ODEProblem(prob_NDE, uvT₀s[i], tspan_train) for i in 1:n_simulations]
    end

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

                @info "loss = $(total_loss): uvT$(profile_percent)% grad$(gradient_percent)% u$(u_percent)% v$(v_percent)% T$(T_percent)% ∂u∂z$(∂u∂z_percent)% ∂v∂z$(∂v∂z_percent)% ∂T∂z$(∂T∂z_percent)% $stage opt$i/$(length(optimizers)) epoch$epoch/$epochs iter$iter/$maxiters"
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

using Oceananigans.Grids: Center, Face

"""
    coarse_grain(Φ, n, ::Type{Center})

Average or coarse grain a `Center`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points and `n` needs to evenly divide `length(Φ)`.
"""
function coarse_grain(Φ, n, ::Type{Center})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

"""
    coarse_grain(Φ, n, ::Type{Face})

Average or coarse grain a `Face`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points. The values at the left and right endpoints of `Φ` will be preserved in the output.
"""
function coarse_grain(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = (N-2) / (n-2)
    Φ̅[1], Φ̅[n] = Φ[1], Φ[N]

    if isinteger(Δ)
        Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Center)
    else
        for i in 2:n-1
            i1 = round(Int, 2 + (i-2)*Δ)
            i2 = round(Int, 2 + (i-1)*Δ)
            Φ̅[i] = mean(Φ[i1:i2])
        end
    end

    return Φ̅
end

"""
    coarse_grain_linear_interpolation(Φ, n, ::Type{Face})

Average or coarse grain a `Face`-centered field `Φ` down to size `n` using linear interpolation. `Φ` is required to have evenly spaced points. The values at the left and right endpoints of `Φ` will be preserved in the output.
"""
function coarse_grain_linear_interpolation(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Φ̅[1] = Φ[1]
    Φ̅[end] = Φ[end]
    gap = (N-1)/(n-1)

    for i=2:n-1
        Φ̅[i] = 1 + (i-1)*gap
    end

    for i=2:n-1
        Φ̅[i] = (floor(Φ̅[i])+1 - Φ̅[i]) * Φ[Int(floor(Φ̅[i]))] + (Φ̅[i] - floor(Φ̅[i])) * Φ[Int(floor(Φ̅[i]))+1]
    end
    return Φ̅
end