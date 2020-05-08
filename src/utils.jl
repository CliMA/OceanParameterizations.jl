# Code credit: https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/utils.jl

"""
    coarse_grain(Φ, n)

Average or coarse grain a field `Φ` down to a size `n`. `Φ` is required to have evenly spaced points and n <= length(Φ).
"""
function coarse_grain(Φ, n)
    m = length(Φ)
    scale = Int(floor(m/n))
    Φ̅ = zeros(n)
    for i in 1:n
        Φ̅[i] = 0
        for j in 1:scale
            Φ̅[i] += Φ[scale*(i-1) + j] / scale
        end
    end
    return Φ̅
end

weights(f::FastDense, θ) = reshape(θ[1:(f.out*f.in)], f.out, f.in)
bias(f::FastDense, θ) = θ[(f.out*f.in+1):end]
