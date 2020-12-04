"""
Constructors for covariance (kernel) functions.

      Constructor                   Description                                Isotropic/Anisotropic
    - SquaredExponentialI(γ,σ):     squared exponential covariance function    isotropic
    - ExponentialI(γ,σ):            exponential covariance function            isotropic
    - RationalQuadraticI(γ,σ,α):    rational quadratic covariance function     isotropic
    - Matern12I(γ,σ):               Matérn covariance function with ʋ = 1/2.   isotropic
    - Matern32I(γ,σ):               Matérn covariance function with ʋ = 3/2.   isotropic
    - Matern52I(γ,σ):               Matérn covariance function with ʋ = 5/2.   isotropic
    - SMP(w,μ,γ):                   Spectral mixture product cov. function

Distance metrics

    - euclidean_distance            l²-norm:  d(x,x') = || x - x' ||",
    - derivative_distance           H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
    - antiderivative_distance       H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
"""
abstract type Kernel end

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | Isotropic covariance functions |
#  *--*--*--*--*--*--*--*--*--*--*--*

""" SquaredExponentialI(γ,σ): squared exponential covariance function, isotropic """
struct SquaredExponentialI <: Kernel
    # Hyperparameters
    # "Length scale"
    γ::Float64
    # "Signal variance"
    σ::Float64
    # "Distance metric"
    d::Function
end

# evaluates the kernel function for a given pair of inputs
function kernel_function(k::SquaredExponentialI; z=nothing)
    # k(x,x') = σ * exp( - d(x,x')² / 2γ² )
    σ = k.σ
    γ = k.γ
    d = k.d
  evaluate(a,b) = σ * exp(- d(a,b,z)^2 / 2*γ^2 )
  return evaluate
end

struct Matern12I <: Kernel
    # Hyperparameters
    "Length scale"
    γ::Float64
    "Signal variance"
    σ::Float64
    "Distance metric"
    d::Function
end

function kernel_function(k::Matern12I; z=nothing)
  # k(x,x') = σ * exp( - ||x-x'|| / γ )
  σ = k.σ
  γ = k.γ
  d = k.d
  evaluate(a,b) = σ * exp(- d(a,b,z) / γ )
  return evaluate
end

struct Matern32I <: Kernel
    # Hyperparameters
    # "Length scale"
    γ::Float64
    # "Signal variance"
    σ::Float64
    # "Distance metric"
    d::Function
end

function kernel_function(k::Matern32I; z=nothing)
    # k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)
    σ = k.σ
    γ = k.γ
    d = k.d
    function evaluate(a,b)
        c = sqrt(3)*d(a,b,z)/γ
        return σ * (1+c) * exp(-c)
    end
  return evaluate
end

struct Matern52I <: Kernel
    # Hyperparameters
    # "Length scale"
    γ::Float64
    # "Signal variance"
    σ::Float64
    # "Distance metric"
    d::Function
end

function kernel_function(k::Matern52I; z=nothing)
    # k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)
    σ = k.σ
    γ = k.γ
    d = k.d
    function evaluate(a,b)
        g = sqrt(5)*d(a,b,z)/γ
        h = 5*(d(a,b,z)^2)/(3*γ^2)
        return σ * (1+g+h) * exp(-g)
    end
  return evaluate
end

struct RationalQuadraticI <: Kernel
    # Hyperparameters
    "Length scale"
    γ::Float64
    "Signal variance"
    σ::Float64
    "Shape parameter"
    α::Float64
    "Distance metric"
    d::Function
end

function kernel_function(k::RationalQuadraticI; z=nothing)
    # k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)
    σ = k.σ
    α = k.α
    d = k.d
    l = γ^2 # squared length scale
    function evaluate(a,b)
       return σ * (1+(a-b)'*(a-b)/(2*α*l))^(-α)
  end
  return evaluate
end

## IN DEVELOPMENT

struct SpectralMixtureProductI <: Kernel
    # """Mixture weights"""
    w::Array{Float64}
    # """Spectral means"""
    μ::Array{Float64}
    # """Spectral variances"""
    γ::Array{Float64}
end

function SpectralMixtureProductI(hyp)
    if length(hyp) % 3 != 0
        throw(error("Inconsistent number of components. Length of parameter vector should be a multiple of three."))
    end

    Q = Int(length(hyp)/3)

    SpectralMixtureProductI(hyp[1:Q],hyp[Q+1:2Q],hyp[2Q+1:3Q])
end

# https://github.com/alshedivat/gpml/blob/master/cov/covSM.m
function kernel_function(k::SpectralMixtureProductI; z=nothing)
    w = (k.w .^2)' # square mixture weights
    μ = k.μ
    γ = k.γ
    h(arg1, arg2) = exp.(-0.5 * arg1) .* cos.(arg2)

    function evaluate(a,b)
        τ = (a .- b) * 2*pi
        D  = length(a)

        K = 1
        for d=1:D
            # println(w)
            # println(h((τ[d] .^ 2)*k.γ, τ[d]*k.μ))
            K *= w * h((τ[d] .^ 2)*γ, τ[d]*μ)
        end
        K
    end
   return evaluate
end

struct SpectralMixtureProductA <: Kernel
    # """Mixture weights"""
    w::Array{Float64} # D x Q array
    # """Spectral means"""
    μ::Array{Float64} # D x Q array
    # """Spectral variances"""
    γ::Array{Float64} # D x Q array
end

function SpectralMixtureProductA(hyp, D)
    Q = Int(floor(length(hyp)/(3D)))
    w = reshape(  hyp[                      1:D*Q],D,Q);   # mixture weights
    μ = reshape(  hyp[D .* Q .+           (1:D*Q)],D,Q);   # spectral means
    γ = reshape(  hyp[D .* Q .+ D .* Q .+ (1:D*Q)],D,Q);   # spectral variances
    SpectralMixtureProductA(w, μ, γ)
end

# https://github.com/alshedivat/gpml/blob/master/cov/covSM.m
function kernel_function(k::SpectralMixtureProductA; z=nothing)
    w = (k.w .^2)' # square mixture weights
    μ = k.μ
    γ = k.γ
    h(arg1, arg2) = exp.(-0.5 * arg1).*cos.(arg2);
    D,Q  = size(w)

    function evaluate(a,b)
        τ = (a .- b) * 2*pi

        K = 1
        for d=1:D
            K = K .* (w[d,:]' * h((τ[d] .^ 2)*γ[d,:], τ[d]*μ[d,:]) )
        end
        K
    end
  return evaluate
end
