
"""
Adapted from sandreza/Learning/sandbox/oceananigans_converter.jl
https://github.com/sandreza/Learning/blob/master/sandbox/oceananigans_converter.jl
"""

using JLD2

struct LESbraryData{𝒮, 𝒯, 𝒰, 𝒱}
    # Initial conditions
    T⁰ :: 𝒮
    U⁰ :: 𝒮
    V⁰ :: 𝒮

    # Fields at each moment in time
    T :: 𝒯
    U :: 𝒯
    V :: 𝒯

    # Some second order statistics at each moment in time
    wT :: 𝒯
    wu :: 𝒯
    wv :: 𝒯
    uu :: 𝒯
    vv :: 𝒯
    ww :: 𝒯

    # Simulation constants
    α  :: 𝒰
    β  :: 𝒰
    f⁰ :: 𝒰
     g :: 𝒰
     L :: 𝒰

    # Time and grid
    t  :: 𝒮
    zC :: 𝒮
    zF :: 𝒮

    # Info about the simulation
    info :: 𝒱
end

function ReadJLD2_LESbraryData(filename)
    les_data = jldopen(filename, "r")
    les_keys = keys(les_data)
    timeseries_keys = keys(les_data["timeseries"]["t"])

    # hold the entries for easy constructor creation
    container = []

    # size of arrays
    Nz = length(collect(les_data["grid"]["zC"])) - 6
    Nt = length(timeseries_keys)

    # Initial Conditions
    T⁰ = zeros(Nz)
    U⁰ = zeros(Nz)
    V⁰ = zeros(Nz)

    # Timeseries
    T = zeros(Nz, Nt)
    U = zeros(Nz, Nt)
    V = zeros(Nz, Nt)
    t = zeros(Nt)

    # Second Order Statistics
    wT  = zeros(Nz+1, Nt)
    uu  = zeros(Nz,   Nt)
    vv  = zeros(Nz,   Nt)
    ww  = zeros(Nz+1, Nt)
    wu  = zeros(Nz+1, Nt)
    wv  = zeros(Nz+1, Nt)

    # grab arrays
    for j in 1:Nt
        key = timeseries_keys[j]

        # Fields
        @. T[:, j] = les_data["timeseries"]["T"][key][1, 1, :]
        @. U[:, j] = les_data["timeseries"]["u"][key][1, 1, :]
        @. V[:, j] = les_data["timeseries"]["v"][key][1, 1, :]

        # Second Order Statistics
        @. wT[:, j] = les_data["timeseries"]["wT"][key][1, 1, :]
        @. wu[:, j] = les_data["timeseries"]["wu"][key][1, 1, :]
        @. wv[:, j] = les_data["timeseries"]["wv"][key][1, 1, :]
        @. uu[:, j] = les_data["timeseries"]["uu"][key][1, 1, :]
        @. vv[:, j] = les_data["timeseries"]["vv"][key][1, 1, :]
        @. ww[:, j] = les_data["timeseries"]["ww"][key][1, 1, :]

        t[j] = les_data["timeseries"]["t"][key]
    end

    # Set initial Conditions
    @. T⁰ = T[:,1]
    @. U⁰ = U[:,1]
    @. V⁰ = V[:,1]

    # Push initial conditions current stuff into container
    push!(container, T⁰, V⁰, U⁰)

    # Push fields into container
    push!(container, T, U, V)

    # Push second order statistics into container
    push!(container, wT, wu, wv, uu, vv, ww)

    # Now grab parameter
    α = les_data["buoyancy"]["equation_of_state"]["α"]
    β = les_data["buoyancy"]["equation_of_state"]["β"]
    f⁰ = les_data["coriolis"]["f"]
    g = les_data["buoyancy"]["gravitational_acceleration"]
    L = les_data["grid"]["Lz"]

    # Push parameters to container
    push!(container, α, β, f⁰, g, L)

    # grab domain data
    zC = collect(les_data["grid"]["zC"])[4:end-3] # padding of 3 on each side
    zF = collect(les_data["grid"]["zF"])[4:end-3] # padding of 3 on each side

    # push
    push!(container, t, zC, zF)

    # Now construct types
    𝒮 = typeof(T⁰)
    𝒯 = typeof(T)
    𝒰 = typeof(α)
    𝒱 = typeof("string")

    # now create data string
    info_string = "The top boundary conditions are flux boundary conditions \n"
    info_string *= "The  bottom boundary condition for temperature is a gradient boundary condition \n"
    info_string *= "The grid data is assumed to be evenly spaced and a power of two \n"

    # push to container
    push!(container, info_string)

    close(les_data)

    return LESbraryData{𝒮, 𝒯, 𝒰, 𝒱}(container...)
end

# avg = "src/les/data/2daySuite/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh128_Nz128_averaged_statistics.jld2"
# les = jldopen(avg, "r")
# keys(les)
# les["timeseries"]

# avg = "src/les/data/general_strat_sandreza/general_strat_4_profiles/general_strat_4_profiles.jld2"

# avg = "src_NDE/les/data/2daySuite/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128/three_layer_constant_fluxes_Qu8.0e-04_Qb5.0e-09_f1.0e-04_Nh128_Nz128_statistics.jld2"
# les_data = jldopen(avg, "r")
# les_keys = keys(les_data)
# timeseries_keys = keys(les_data["timeseries"]["t"])

# for k in keys(les_data["timeseries"])
#     println(k)
# end

# stats = "src/les/data/2daySuite/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_Nh128_Nz128/three_layer_constant_fluxes_Qu1.0e-03_Qb0.0e+00_Nh128_Nz128_statistics.jld2"
# ReadJLD2_OceananigansData(stats)
