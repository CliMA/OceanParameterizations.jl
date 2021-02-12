function local_richardson(uvT, 𝒟; unscale)
    H = Float32(abs(𝒟.uw.z[end] - 𝒟.uw.z[1]))
    g = 9.81f0
    α = 1.67f-4
    u_scaling = 𝒟.scalings["u"]
    v_scaling = 𝒟.scalings["v"]
    T_scaling = 𝒟.scalings["T"]
    σ_u = Float32(u_scaling.σ)
    σ_v = Float32(v_scaling.σ)
    σ_T = Float32(T_scaling.σ)

    Nz = Int(size(uvT, 1) / 3)
    D_cell = Float32.(Dᶜ(Nz, 1 / Nz))
    D_face = Float32.(Dᶠ(Nz, 1 / Nz))
    Ris = similar(uvT, Nz + 1, size(uvT,2))
    for i in 1:size(Ris, 2)
        u = @view uvT[1:Nz, i]
        v = @view uvT[Nz + 1:2Nz, i]
        T = @view uvT[2Nz + 1:3Nz, i]
        if unscale
            u .= u_scaling.(u)
            v .= v_scaling.(v)
            T .= T_scaling.(T)
        end
        Ri = @view Ris[:, i]
        ∂u∂z = D_face * u
        ∂v∂z = D_face * v
        ∂T∂z = D_face * T
        Ri .= (H * g * α * σ_T .* ∂T∂z) ./ ((σ_u .* ∂u∂z) .^2 + (σ_v .* ∂v∂z) .^2)
    end
    
    for i in 1:length(Ris)
        if isnan(Ris[i])
            Ris[i] = 0
        end

        if Ris[i] == Inf
            Ris[i] = 10f10
        end
    end
    return Ris
end