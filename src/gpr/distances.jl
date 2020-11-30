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

# abstract type Distance end
#
# """
# EuclideanDistance: computes the Euclidean distance (l²-norm) between two vectors
# EuclideanDistance(x,x') = || x - x' ||
# """
# mutable struct EuclideanDistance
#     zavg::Array
#     EuclideanDistance() = EuclideanDistance
#     EuclideanDistance(zavg) = new(zavg)
#     EuclideanDistance(a,b) = sqrt(sq_mag(a,b))
# end
#
# """
# DerivativeDistance: computes the H¹-norm w.r.t z of two vectors
# DerivativeDistance(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||
# """
# Base.@kwdef struct DerivativeDistance
#     zavg::Array
#     DerivativeDistance() = DerivativeDistance
#     DerivativeDistance(zavg) = new(zavg)
#     DerivativeDistance(a,b) = sqrt(sq_mag( δ(a, zavg), δ(b, zavg) ))
# end
#
# """
# AntiderivativeDistance: computes the H⁻¹-norm w.r.t z of two vectors
# AntiderivativeDistance(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||
# """
# Base.@kwdef struct AntiderivativeDistance
#     zavg::Array
#     AntiderivativeDistance() = AntiderivativeDistance
#     AntiderivativeDistance(zavg) = new(zavg)
#     AntiderivativeDistance(a,b) = sqrt(sq_mag((diff(a).*diff(zavg) , diff(b).*diff(zavg))))
# end

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

# """
# l2norm_strat_penalty: computes the Euclidean distance (l²-norm) between two vectors and adds
# the a proxy for the difference in the initial stratification from the corresponding temperature
# profiles, as approximated by the stratification at the bottom.
# """
# function l2norm_strat_penalty(a,b,z) # d(x,x') = || x - x' ||
#     α_proxy(x) = x[2] - x[1]
#     println(abs(α_proxy(a)-α_proxy(b)))
#     if abs(α_proxy(a)-α_proxy(b))>0.05
#         return l2_norm(a,b) + 0.0001
#     end
#     return l2_norm(a,b)
# end
