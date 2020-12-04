
using OceanParameterizations, Plots

𝒟 = data("strong_wind")

<<<<<<< refs/remotes/origin/ali/bit-of-cleanup:wind_mixing/test_reconstruct_fluxes.jl
𝒟_reconstructed = data("strong_wind_weak_heating", reconstruct_fluxes=true)
=======
    Nz,Nt = size(T)

    dudt = (u[:,2:Nt] .- u[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dVdt values
    dvdt = (v[:,2:Nt] .- v[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dVdt values
    dTdt = (T[:,2:Nt] .- T[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dVdt values
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

    duw_dz = -dudt .+ f*v
    dvw_dz = -dvdt .- f*u
    dwT_dz = -dTdt

    # println(size(wV(duw_dz)))
    # u, v, T, uw, vw, wT, t
    return (u, v, T, wϕ(duw_dz), wϕ(dvw_dz), wϕ(dwT_dz), t[1:Nt-1])
end

using ClimateParameterizations, Plots
#
𝒟 = ClimateParameterizations.Data.data("strong_wind")

𝒟_reconstructed = ClimateParameterizations.Data.data("strong_wind_weak_heating", reconstruct_fluxes=true)
>>>>>>> Correct DE's in reconstruct_flux_profiles function:src/data/reconstruct_fluxes.jl

𝒟_reconstructed
z = 𝒟_reconstructed.uw.z
t = 𝒟_reconstructed.t
Nt = length(𝒟.t)
output_gif_directory = "TestReconstructFluxes"
animate_gif((𝒟_reconstructed.uw.coarse, 𝒟.uw.coarse[:,1:Nt-1]), z, t, "uw",
            x_label=["∫(-du/dt + fv)dz", "truth"],
            filename="uw_reconstructed",
            directory=output_gif_directory)
animate_gif((𝒟_reconstructed.vw.coarse, 𝒟.vw.coarse[:,1:Nt-1]), z, t, "vw",
            x_label=["∫(-dv/dt - fu)dz", "truth"],
            filename="vw_reconstructed",
            directory=output_gif_directory)
animate_gif((𝒟_reconstructed.wT.coarse, 𝒟.wT.coarse[:,1:Nt-1]), z, t, "wT",
            x_label=["∫(-dw/dt)dz", "truth"],
            filename="wT_reconstructed",
            directory=output_gif_directory)
