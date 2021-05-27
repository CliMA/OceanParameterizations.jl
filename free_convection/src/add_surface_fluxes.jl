function add_surface_fluxes!(ds)
    Qθ = ds.metadata["temperature_flux"]

    T = ds["T"]
    wT = ds["wT"]

    _, _, Nz, _ = size(T)
    wT[:, :, Nz+1, :] .= Qθ

    return ds
end
