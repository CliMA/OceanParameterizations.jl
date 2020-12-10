# Distance metrics for kernel functions

δ(Φ, z) = diff(Φ) ./ diff(z)

"""
    sq_mag(a, b)

||a - b||^2 = norm(a - b)^2 but more efficient.
"""
function sq_mag(a, b)
    ll = 0.0
    @inbounds for k in 1:length(a)
        ll += (a[k] - b[k])^2
    end
    return ll
end

"""
    euclidean_distance(a, b, z)

Computes the Euclidean distance (l²-norm) between two vectors:

    d(x, x') = || x - x' ||
"""
@inline euclidean_distance(a, b, z) = sqrt(sq_mag(a, b))

@inline euclidean_distance(a, b) = sqrt(sq_mag(a, b))

"""
    derivative_distance(a, b, z)

Computes the H¹-norm with respect to z of two vectors:

    d(x, x') = || diff(x) / diff(z) - diff(x') / diff(z) ||
"""
@inline derivative_distance(a, b, z) = sqrt(sq_mag( δ(a, z), δ(b, z) ))

"""
    antiderivative_distance(a, b, z)

Computes the H⁻¹-norm with respect to z of two vectors:

    d(x, x′) = || diff(x) * diff(z) - diff(x') * diff(z) ||
"""
@inline antiderivative_distance(a,b,z) = sqrt(sq_mag(diff(a) .* diff(z), diff(b) .* diff(z)))
