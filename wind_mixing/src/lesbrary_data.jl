
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

    # Subfilter fluxes
    νₑ_∂z_u :: 𝒯
    νₑ_∂z_v :: 𝒯
    νₑ_∂z_w :: 𝒯
    κₑ_∂z_T :: 𝒯

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

    # Boundary conditions
    θ_top    :: 𝒰
    u_top    :: 𝒰
    θ_bottom :: 𝒰
    u_bottom :: 𝒰

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

    # Subfilter momentum fluxes
    νₑ_∂z_u  = zeros(Nz+1, Nt)
    νₑ_∂z_v  = zeros(Nz+1, Nt)
    νₑ_∂z_w  = zeros(Nz, Nt)
    κₑ_∂z_T  = zeros(Nz+1, Nt)

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

        # Subfilter fluxes
        @. νₑ_∂z_u[:, j] = les_data["timeseries"]["νₑ_∂z_u"][key][1, 1, :]
        @. νₑ_∂z_v[:, j] = les_data["timeseries"]["νₑ_∂z_v"][key][1, 1, :]
        @. νₑ_∂z_w[:, j] = les_data["timeseries"]["νₑ_∂z_w"][key][1, 1, :]
        @. κₑ_∂z_T[:, j] = les_data["timeseries"]["κₑ_∂z_T"][key][1, 1, :]

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

    # Push subfilter fluxes into container
    push!(container, νₑ_∂z_u, νₑ_∂z_v, νₑ_∂z_w, κₑ_∂z_T)

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

    # now grab boundary condition data
    θ_top = les_data["boundary_conditions"]["θ_top"]
    u_top = les_data["boundary_conditions"]["u_top"]
    θ_bottom = les_data["boundary_conditions"]["θ_bottom"]
    u_bottom = les_data["boundary_conditions"]["u_bottom"]

    # push to container
    push!(container, θ_top, u_top, θ_bottom, u_bottom)

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

# TESTING
# file = "2DaySuite/three_layer_constant_fluxes_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection_statistics.jld2"
# les = jldopen(file, "r")
# keys(les["timeseries"])
# les["timeseries"]["κₑ_∂z_T"]
# les["timeseries"]["v"]["0"][1,1,:]
# les["grid"]["Δx"]

# using Plots
# file = "2DaySuite/three_layer_constant_fluxes_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection_statistics.jld2"
# myles = ReadJLD2_LESbraryData(file)
#
# vbl = myles.κₑ_∂z_T
# xlims=(-1.2e-5, maximum(myles.κₑ_∂z_T))
# vbl = myles.wT
# xlims=(minimum(myles.wT), maximum(myles.wT))
#
# anim = @animate for i=1:288
#     plot(vbl[:,i], myles.zF, legend=false, xlims=xlims, size=(400,600))
# end
# gif(anim, "wT.gif")
#
# Δt = diff(myles.t, dims=1)'
# Nz,Nt = size(myles.T)
# dudt = (myles.u[:,2:Nt] .- myles.u[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dUdt values
