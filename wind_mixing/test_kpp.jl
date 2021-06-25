using WindMixing
using OceanParameterizations
using OceanTurb

train_files = ["wind_-3.5e-4_diurnal_3.5e-8"]

𝒟test = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

constants = (f=1f-4, α=2f-4, g=9.80655f0, Nz=32, H=256)

wT_flux_top = diurnal_fluxes(train_files, constants)[1]

BCs_unscaled = (uw=(top=𝒟test.uw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    vw=(top=𝒟test.vw.coarse[end, 1], bottom=𝒟test.uw.coarse[1, 1]), 
    wT=(top=wT_flux_top, bottom=𝒟test.wT.coarse[1, 1]))
    
ICs_unscaled = (u=𝒟test.u.coarse[:,1], v=𝒟test.v.coarse[:,1], T=𝒟test.T.coarse[:,1])

t = 𝒟test.t[1:1:1153]

sol_kpp = column_model_1D_kpp(constants, BCs_unscaled, ICs_unscaled, t, OceanTurb.KPP.Parameters())