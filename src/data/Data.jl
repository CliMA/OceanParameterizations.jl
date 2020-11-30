"""
Data module for preparing data.
"""

module Data

using OceanTurb
using OrderedCollections
using Statistics

include("../data/coarse_graining.jl")
export coarse_grain

# harvesting Oceananigans data
include("../les/read_les_output.jl")
export read_les_output

include("feature_scaling.jl")
export  AbstractFeatureScaling,
        ZeroMeanUnitVarianceScaling,
        MinMaxScaling,
        scale,
        unscale

include("modify_predictor_fns.jl")
export append_tke,
       partial_temp_profile

include("convective_adjust.jl")
export convective_adjust!

# running OceanTurb KPP simulations based on OceananigansData conditions
include("../kpp/run.jl")
export closure_free_convection_kpp_full_evolution,
       closure_free_convection_kpp

# running OceanTurb TKE simulations based on OceananigansData conditions
include("../tke/run.jl")
export closure_free_convection_tke_full_evolution,
       closure_free_convection_tke

# Structs
export  ProfileData,
        VData

# Functions
export  data,
        animate_gif

struct VData # for each of uw, vw, and wT
    z # z vector for the variable
    coarse # Nz x Nt array of unscaled profiles
    scaled # Nz x Nt array of scaled profiles
    unscale_fn # function to unscaled profile vectors with
    training_data # (uvT, scaled) pairs
end

struct ProfileData
    grid_points # Integer
    uvT_scaled # 3Nz x Nt array
    u_scaled # Nz x Nt array
    v_scaled # Nz x Nt array
    T_scaled # Nz x Nt array
    T_coarse
    uw::VData
    vw::VData
    wT::VData
    t # timeseries Vector
    scalings::Dict # maps name (e.g. "uw") to the AbstractFeatureScaling object associated with it
end

"""
data(filename::String; animate=false, scale_type::AbstractScaling=ZeroMeanUnitVarianceScaling
# Arguments
- filename: (String)            "free_convection"
- animate: (Bool)               Whether to save an animation of all the original profiles
- animate_dir: (Bool)           Directory to save the animation files to
- scale_type: (AbstractFeatureScaling) ZeroMeanUnitVarianceScaling or MinMaxScaling
- override_scalings: (Dict)     For if you want the testing simulation data to be scaled in the same way as the training data. Set to 𝒟train.scalings to use the scalings from 𝒟train.
"""
function data(filenames; animate=false, scale_type=MinMaxScaling, animate_dir="Output", override_scalings::Dict=nothing)

    if typeof(filenames) <: String; filenames = [filenames] end

    # harvest data from Oceananigans simulation output files
    all_les = Dict()
    for file in filenames
        all_les[file] = read_les_output(file) # <: LESbraryData
    end
    get_array(f) = cat((f(les) for (file,les) in all_les)...,dims=2)

    uw = get_array(les -> les.wu)
    vw = get_array(les -> les.wv)
    wT = get_array(les -> les.wT)
    u  = get_array(les -> les.U)
    v  = get_array(les -> les.V)
    T  = get_array(les -> les.T)
    t  = get_array(les -> les.t)

    zC = all_les[filenames[1]].zC
    zF = all_les[filenames[1]].zF

    if animate == true
        plot(T[:,end], zC)
        animate_gif([uw], zF, t, "uw", dir=animate_dir)
        animate_gif([vw], zF, t, "vw", dir=animate_dir)
        animate_gif([wT], zF, t, "wT", dir=animate_dir)
        animate_gif([u], zC, t, "u", dir=animate_dir)
        animate_gif([v], zC, t, "v", dir=animate_dir)
        animate_gif([T], zC, t, "T", dir=animate_dir)
    end

    coarsify_cell(x) = cat((coarse_grain(x[:,i], 32, Cell) for i in 1:size(x,2))..., dims=2)
    coarsify_face(x) = cat((coarse_grain(x[:,i], 33, Face) for i in 1:size(x,2))..., dims=2)
    u_coarse = coarsify_cell(u)
    v_coarse = coarsify_cell(v)
    T_coarse = coarsify_cell(T)
    uw_coarse = coarsify_face(uw)
    vw_coarse = coarsify_face(vw)
    wT_coarse = coarsify_face(wT)

    zC_coarse = coarse_grain(zC, 32, Cell)
    zF_coarse = coarse_grain(zF, 33, Face)
    # zF_coarse = coarse_grain_linear_interpolation(zF, 33, Face)

    if override_scalings==nothing
        # set the scaling according to the data (for training simulations)
        get_scaling(coarse) = scale_type(coarse)
    else
        # for if you want the testing simulation data to be scaled in the same way as the training data
        get_scaling(coarse) = override_scalings[coarse]
    end

    all_scalings=Dict()
    for (name, coarse) in [("u",u_coarse), ("v",v_coarse), ("T",T_coarse),
                         ("uw",uw_coarse), ("vw",vw_coarse), ("wT",wT_coarse)]
        all_scalings[name] = get_scaling(coarse)
    end

    get_scaled(name, coarse) = all_scalings[name].(coarse)
    u_scaled = get_scaled("u", u_coarse)
    v_scaled = get_scaled("v", v_coarse)
    T_scaled = get_scaled("T", T_coarse)
    uvT_scaled = cat(u_scaled, v_scaled, T_scaled, dims=1)

    function get_VData(name, coarse, z)
        scaling = get_scaling(coarse)
        scaled = get_scaled(name, coarse)
        # unscale_fn(x) = unscale(x, scaling)
        unscale_fn = Base.inv(scaling)
        training_data = [(uvT_scaled[:,i], scaled[:,i]) for i in 1:size(uvT_scaled,2)] # (predictor, target) pairs
        return VData(z, coarse, scaled, unscale_fn, training_data)
    end

    uw = get_VData("uw", uw_coarse, zF_coarse)
    vw = get_VData("vw", vw_coarse, zF_coarse)
    wT = get_VData("wT", wT_coarse, zF_coarse)

    return ProfileData(33, uvT_scaled, u_scaled, v_scaled, T_scaled, T_coarse, uw, vw, wT, t, all_scalings)
end

end
