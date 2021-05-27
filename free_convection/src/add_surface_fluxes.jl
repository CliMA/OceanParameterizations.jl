function add_surface_fluxes!(ds)
    Qθ = ds.metadata["temperature_flux"]

    T = ds["T"]
    wT = ds["wT"]

    Nz = size(T, 3)
    wT[:, :, Nz+1, :] .= Qθ

    return ds
end
