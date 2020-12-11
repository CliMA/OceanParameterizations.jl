
using OceanParameterizations, Plots

𝒟 = data("strong_wind", reconstruct_fluxes=false)
𝒟_reconstructed = data("strong_wind", reconstruct_fluxes=true)
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
