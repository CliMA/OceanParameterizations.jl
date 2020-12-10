# # Old
# directories = Dict(
#     "free_convection"          => "2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_statistics.jld2",
#     "strong_wind"              => "2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh128_Nz128_statistics.jld2",
#     "strong_wind_no_coriolis"  => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh128_Nz128_statistics.jld2",
#     "weak_wind_strong_cooling" => "2daySuite/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu2.0e-04_Qb1.0e-07_f1.0e-04_Nh128_Nz128_statistics.jld2",
#     "strong_wind_weak_cooling" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_statistics.jld2",
#     "strong_wind_weak_heating" => "2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb-1.0e-08_f1.0e-04_Nh128_Nz128_statistics.jld2",
# )

# New
directories = Dict(
    "free_convection"          => "2DaySuite/three_layer_constant_fluxes_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection_statistics.jld2",
    "strong_wind"              => "2DaySuite/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind_statistics.jld2",
    "strong_wind_no_coriolis"  => "2DaySuite/three_layer_constant_fluxes_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation_statistics.jld2",
    "weak_wind_strong_cooling" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu3.0e-04_Qb1.0e-07_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling_statistics.jld2",
    "strong_wind_weak_cooling" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu8.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling_statistics.jld2",
    "strong_wind_weak_heating" => "2DaySuite/three_layer_constant_fluxes_hr48_Qu1.0e-03_Qb-4.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_heating_statistics.jld2"
)

function read_les_output(filename::String)
    filename = pwd() * "/" * directories[filename]
    return ReadJLD2_LESbraryData(filename)
end

"""
# Description
Takes NzxNt arrays of profiles for variables u, v, T and returns
Nzx(Nt-1) arrays of the profile evolution for u, v, T, u'w', v'w', and w'T'
the horizontally averaged flux for variable V.

# Arguments
Unscaled u, v, T, z, t, and f
"""
function reconstruct_flux_profiles(u, v, T, νₑ_∂z_u, νₑ_∂z_v, κₑ_∂z_T, z, t, f)

    Δz = diff(z)
    Δt = diff(t, dims=1)'

    Nz,Nt = size(T)

    dudt = (u[:,2:Nt] .- u[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dUdt values
    dvdt = (v[:,2:Nt] .- v[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dVdt values
    dTdt = (T[:,2:Nt] .- T[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dTdt values

    vₑ_∂²z_u = (vₑ_∂z_u[1:Nz-1,:] .- vₑ_∂z_u[2:Nz,:]) ./ Δz
    vₑ_∂²z_v = (vₑ_∂z_v[1:Nz-1,:] .- vₑ_∂z_v[2:Nz,:]) ./ Δz
    κₑ_∂²z_T = (κₑ_∂z_T[1:Nz-1,:] .- κₑ_∂z_T[2:Nz,:]) ./ Δz

    u = u[:,1:Nt-1]
    v = v[:,1:Nt-1]
    T = T[:,1:Nt-1]

    """ evaluates wϕ = ∫ ∂z(wϕ) dz """
    function wϕ(∂z_wϕ)
        ans = zeros(Nz+1, Nt-1) # one fewer column than T
        for i in 1:Nt-1, h in 1:Nz-1
            ans[h+1, i] = ans[h, i] + Δz[h] * ∂z_wϕ[h, i]
        end
        return ans
    end

    duw_dz = -dudt .+ f*v .+ vₑ_∂²z_u
    dvw_dz = -dvdt .- f*u .+ vₑ_∂²z_v
    dwT_dz = -dTdt .+ κₑ_∂²z_T

    # u, v, T, uw, vw, wT, t
    return (u, v, T, wϕ(duw_dz), wϕ(dvw_dz), wϕ(dwT_dz), t[1:Nt-1])
end

struct FluxData{Z, C, S, U, T} # for each of uw, vw, and wT
                z :: Z # z vector for the variable
           coarse :: C # Nz x Nt array of unscaled profiles
           scaled :: S # Nz x Nt array of scaled profiles
       unscale_fn :: U # function to unscaled profile vectors with
    training_data :: T # (uvT, scaled) pairs
end

struct uvTData{Z, C, S, U} # for each of u, v, T
             z :: Z # z vector for the variable
        coarse :: C # Nz x Nt array of unscaled profiles
        scaled :: S # Nz x Nt array of scaled profiles
    unscale_fn :: U # function to unscaled profile vectors with
end

struct ProfileData{Σ, U, V, Θ, UW, VW, WT, T, D}
    grid_points :: Int
     uvT_scaled :: Σ  # 3Nz x Nt array
              u :: U
              v :: V
              T :: Θ
             uw :: UW
             vw :: VW
             wT :: WT
              t :: T  # timeseries Vector
       scalings :: D  # Dict mapping names (e.g. "uw") to the AbstractFeatureScaling object associated with it.
end

"""
    data(filenames; animate=false, scale_type=MinMaxScaling, animate_dir="Output", override_scalings=nothing, reconstruct_fluxes=false)

# Arguments
- filenames                "free_convection"
- animate                  Whether to save an animation of all the original profiles
- animate_dir              Directory to save the animation files to
- scale_type               ZeroMeanUnitVarianceScaling or MinMaxScaling
- override_scalings::Dict  For if you want the testing simulation data to be scaled in the same way as the training data.
                           Set to 𝒟train.scalings to use the scalings from 𝒟train.
"""
function data(filenames; animate=false, scale_type=MinMaxScaling, animate_dir="Output", override_scalings=nothing, reconstruct_fluxes=false)

    filenames isa String && (filenames = [filenames])

    # Harvest data from Oceananigans simulation output files.
    all_les = Dict()

    for file in filenames
        all_les[file] = read_les_output(file)
    end

    get_array(f) = cat((f(les) for (file, les) in all_les)..., dims=2)

    uw = get_array(les -> les.wu)
    vw = get_array(les -> les.wv)
    wT = get_array(les -> les.wT)
    u  = get_array(les -> les.U)
    v  = get_array(les -> les.V)
    T  = get_array(les -> les.T)
    t  = get_array(les -> les.t)
    νₑ_∂z_u = get_array(les -> les.νₑ_∂z_u)
    νₑ_∂z_v = get_array(les -> les.νₑ_∂z_v)
    κₑ_∂z_T = get_array(les -> les.κₑ_∂z_T)

    first = all_les[filenames[1]]
    zC = first.zC
    zF = first.zF

    if animate
        animate_gif([uw], zF, t, "uw", directory=animate_dir)
        animate_gif([vw], zF, t, "vw", directory=animate_dir)
        animate_gif([wT], zF, t, "wT", directory=animate_dir)
        animate_gif([u],  zC, t, "u",  directory=animate_dir)
        animate_gif([v],  zC, t, "v",  directory=animate_dir)
        animate_gif([T],  zC, t, "T",  directory=animate_dir)
    end

    coarsify_cell(x) = cat((coarse_grain(x[:,i], 32, Cell) for i in 1:size(x,2))..., dims=2)
    coarsify_face(x) = cat((coarse_grain(x[:,i], 33, Face) for i in 1:size(x,2))..., dims=2)

    u_coarse  = coarsify_cell(u)
    v_coarse  = coarsify_cell(v)
    T_coarse  = coarsify_cell(T)
    uw_coarse = coarsify_face(uw)
    vw_coarse = coarsify_face(vw)
    wT_coarse = coarsify_face(wT)
    νₑ_∂z_u   = coarsify_face(νₑ_∂z_u)
    νₑ_∂z_v   = coarsify_face(νₑ_∂z_v)
    κₑ_∂z_T   = coarsify_face(κₑ_∂z_T)

    zC_coarse = coarse_grain(zC, 32, Cell)
    zF_coarse = coarse_grain(zF, 33, Face)

    f = 1e-4 # for now

    if reconstruct_fluxes
        u_coarse, v_coarse, T_coarse, uw_coarse, vw_coarse, wT_coarse, t =
            reconstruct_flux_profiles(u_coarse, v_coarse, T_coarse, νₑ_∂z_u, νₑ_∂z_v, κₑ_∂z_T, zF, t, f)
    end

    function get_scaling(name, coarse)
        if isnothing(override_scalings)
            # set the scaling according to the data (for training simulations)
            return scale_type(coarse)
        else
            # for if you want the testing simulation data to be scaled in the same way as the training data
            return override_scalings[name]
        end
    end

    all_scalings=Dict()

    for (name, coarse) in [("u", u_coarse), ("v", v_coarse), ("T", T_coarse),
                           ("uw", uw_coarse), ("vw", vw_coarse), ("wT", wT_coarse)]
        all_scalings[name] = get_scaling(name, coarse)
    end

    function get_scaled(name, coarse)
        scaling = get_scaling(name, coarse)
        scaled = all_scalings[name].(coarse)
        unscale_fn = Base.inv(scaling)
        return (scaled, unscale_fn)
    end

    function get_uvTData(name, coarse, z)
        scaled, unscale_fn = get_scaled(name, coarse)
        return uvTData(z, coarse, scaled, unscale_fn)
    end

    function get_FluxData(name, coarse, z)
        scaled, unscale_fn = get_scaled(name, coarse)
        training_data = [(uvT_scaled[:,i], scaled[:,i]) for i in 1:size(uvT_scaled,2)] # (predictor, target) pairs
        return FluxData(z, coarse, scaled, unscale_fn, training_data)
    end

    u = get_uvTData("u", u_coarse, zC_coarse)
    v = get_uvTData("v", v_coarse, zC_coarse)
    T = get_uvTData("T", T_coarse, zC_coarse)
    uvT_scaled = cat(u.scaled, v.scaled, T.scaled, dims=1)

    uw = get_FluxData("uw", uw_coarse, zF_coarse)
    vw = get_FluxData("vw", vw_coarse, zF_coarse)
    wT = get_FluxData("wT", wT_coarse, zF_coarse)

    return ProfileData(33, uvT_scaled, u, v, T, uw, vw, wT, t, all_scalings)
end
