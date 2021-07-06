function convective_adjustment_flux(T::Field, K)
    grid = T.grid
    Nz, Δz = grid.Nz, grid.Δz

    ∂T∂z = ComputedField(@at (Center, Center, Face) ∂z(T))
    compute!(∂T∂z)

    κ_∂T∂z = ZFaceField(grid)

    # Convective adjustment only acts in the interior so we ignore k=1 and k=Nz+1.
    for k in 2:Nz
        κ = ∂T∂z[1, 1, k] < 0 ? K : 0
        κ_∂T∂z[1, 1, k] = κ * ∂T∂z[1, 1, k]
    end

    return κ_∂T∂z
end

function convective_adjustment_flux(T::FieldTimeSeries, K)
    κ_∂T∂z = FieldTimeSeries(T.grid, (Center, Center, Face), T.times, ArrayType=Array{Float32})

    Nt = size(κ_∂T∂z, 4)
    for n in 1:Nt
        κ_∂T∂z_n = convective_adjustment_flux(T[n], K)
        κ_∂T∂z.data[:, :, :, n] .= κ_∂T∂z_n.data
    end

    return κ_∂T∂z
end

function add_convective_adjustment_flux!(ds, K)
    ds.fields["wT_param"] = convective_adjustment_flux(ds.fields["T"], K)
    return nothing
end
