# Distance metrics for kernel functions

δ(Φ, z) = diff(Φ) ./ diff(z)

# this is norm(a-b)^2 but more efficient
function sq_mag(a,b) # ||a - b||^2
    ll = 0.0
    @inbounds for k in 1:length(a)
        ll += (a[k]-b[k])^2
    end
    return ll
end

"""
euclidean_distance: computes the Euclidean distance (l²-norm) between two vectors
"""
@inline function euclidean_distance(a,b,z) # d(x,x') = || x - x' ||
    return sqrt(sq_mag(a,b))
end

@inline function euclidean_distance(a,b) # d(x,x') = || x - x' ||
    return sqrt(sq_mag(a,b))
end

"""
derivative_distance: computes the H¹-norm w.r.t z of two vectors
"""
function derivative_distance(a,b,z) # d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||
    return sqrt(sq_mag( δ(a, z), δ(b, z) ))
end

"""
antiderivative_distance: computes the H⁻¹-norm w.r.t z of two vectors
"""
function antiderivative_distance(a,b,z) # || diff(x).*diff(z) - diff(x').*diff(z) ||
    return sqrt(sq_mag(diff(a).*diff(z) , diff(b).*diff(z)))
end
