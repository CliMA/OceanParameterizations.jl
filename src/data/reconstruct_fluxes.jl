
"""
# Description
Takes NzxNt arrays of profiles for variables u, v, T and returns
Nzx(Nt-1) arrays of the profile evolution for u, v, T, u'w', v'w', and w'T'
the horizontally averaged flux for variable V.

# Arguments
Unscaled u, v, T, z, t, and f
"""
function reconstruct_flux_profiles(u, v, T, z, t, f)

    Δz = diff(z)
    Δt = diff(t, dims=1)'

    Nz,Nt = size(T)

    dVdt = (T[:,2:Nt] .- T[:,1:Nt-1]) ./ Δt # Nz x (Nt-1) array of approximate dVdt values
    u = u[:,1:Nt-1]
    v = v[:,1:Nt-1]
    T = T[:,1:Nt-1]

    function wV(dwV_dz)
        """ evaluates wV = ∫(dwV_dz)dz
        """
        ans = zeros(Nz+1, Nt-1) # one fewer column than T
        for i=1:Nt-1
            # ans[1,i] = 0.0
            # ans[2,i] = Δz[1]*dwV_dz[1,i]
            for h=1:Nz-1
                c = 0.5*Δz[h]*(dwV_dz[h+1,i]+dwV_dz[h,i]) # trapezoidal riemann sum
                ans[h+1,i] = ans[h,i] + c
            end
        end
        ans
    end

    duw_dz = -dVdt .+ f*v
    dvw_dz = -dVdt .- f*u
    dwT_dz = -dVdt

    # println(size(wV(duw_dz)))
    # u, v, T, uw, vw, wT, t
    return (u, v, T, wV(duw_dz), wV(dvw_dz), wV(dwT_dz), t[1:Nt-1])
end

using ClimateParameterizations, Plots
#
𝒟 = ClimateParameterizations.Data.data("strong_wind")

𝒟_reconstructed = ClimateParameterizations.Data.data("strong_wind_weak_heating", reconstruct_fluxes=true)

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
