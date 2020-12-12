function add_surface_heat_fluxes(A)
    Qθ = A.metadata[:_stack][:heat_flux]
    A_data = A.data
    A_data[end, :] .= Qθ
    return GeoArray(A_data, dims=dims(A), name=GeoData.name(A), refdims=refdims(A), metadata=metadata(A), missingval=missingval(A))
end

function add_surface_fluxes(ds)
    vars = keys(ds)
    layers = [var == :wT ? add_surface_heat_fluxes(ds[var]) : ds[var] for var in vars]
    return GeoStack(layers..., keys=vars, window=window(ds), refdims=refdims(ds), metadata=metadata(ds))
end
