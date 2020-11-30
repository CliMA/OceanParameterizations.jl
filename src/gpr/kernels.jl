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

##

# """
# https://github.com/alshedivat/gpml/blob/master/cov/covSM.m
#
# hyp = [ log(w(:))
#          log(m(:))
#          log(sqrt(v(:))) ]
#
# Q: number of components
# D: dimension of the input
#
# """

#

# function 🍮(🍨, 🍳, 😎)
#     🍨 + 🍳+ 😎
# end

##
# function SMP(Q,hyp,a,b)
#       D = length(a);
#       n = 1                                                   # dimensionality
#       w = reshape(  hyp[                     1:D*Q],D,Q);     # mixture weights
#       m = reshape(  hyp[D .* Q .+           (1:D*Q)],D,Q);    # spectral means
#       v = reshape(2*hyp[D .* Q .+ D .* Q .+ (1:D*Q)],D,Q);    # spectral variances
#
#       T = 2*pi*bsxfun(@minus,reshape(a,n,1,D),reshape(b,1,[],D));
#
#       T = reshape(T,[],D);
#
#       K = 1;
#       w = reshape(w,Q,D)';
#       m = reshape(m,Q,D)';
#       v = reshape(v,Q,D)';
#
#       h(t2v, tm) = exp(-0.5 * t2v).*cos(tm);
#       for d=1:D
#           K = K .* ( h( (T[:,d] .* T[:,d]) * v[d,:], T(:,d)*m(d,:) )*w[d,:]' );
#       end
#       K = reshape(K.*ones(size(T,1),1),n,[]);
#
#       return K
# end

# function covSMfast(Q,hyp,x,z)
#     smp = Q<0
#     Q = abs(Q)
#     n,D = size(x); P = smp*D+(1-smp);               # dimensionality, P=D or P=1
#     w = exp(reshape(  hyp(         1:P*Q) ,P,Q));     # mixture weights
#     m = exp(reshape(  hyp(P*Q+    (1:D*Q)),D,Q));     # spectral means
#     v = exp(reshape(2*hyp(P*Q+D*Q+(1:D*Q)),D,Q));     # spectral variances
#
#     T = reshape(T,[],D);
#
#     if smp
#         h(t2v, tm) = exp(-0.5*t2v).*cos(tm);
#         K = 1; w = reshape(w,Q,P)'; m = reshape(m,Q,D)'; v = reshape(v,Q,D)';
#         for d=1:D
#             K = K .* ( h( (T(:,d).*T(:,d))*v(d,:), T(:,d)*m(d,:) )*w(d,:)' );
#     end
#     K = reshape(K.*ones(size(T,1),1),n,[]);
#     else
         # E = exp(-0.5*(T.*T)*v); H = E.*cos(T*m);
         # K = reshape(H*w',n,[]);
         # return K
#   end
# end
#
#
#
# function SM(Q,hyp,x,z)
#   Q = abs(Q)
#   n,D = size(x);                                  # dimensionality
#   w = exp(reshape(  hyp(         1:Q) ,1,Q));     # mixture weights
#   m = exp(reshape(  hyp(Q+    (1:D*Q)),D,Q));     # spectral means
#   v = exp(reshape(2*hyp(Q+D*Q+(1:D*Q)),D,Q));     # spectral variances
#
#   # w = exp(reshape(  hyp(         1) ,1,Q));     # mixture weights
#   # m = exp(reshape(  hyp(1+    (1:D)),D,Q));     # spectral means
#   # v = exp(reshape(2*hyp(1+16+ (1:D)),D,Q));     # spectral variances
#
#   T = reshape(T,[],D);
#   E = exp(-0.5*(T.*T)*v);
#   H = E.*cos(T*m);
#   K = reshape(H*w',n,[]);
#   return K
# end
#
#

# function SMP(hyp,a,b)
#
#       τ = (a .- b) * 2*pi
#
#       D = length(a);
#       Q = Int(length(hyp)/(3*D))
#
#       println(Q)
#
#       w .^= 2 # square mixture weights
#
#       h(arg1, arg2) = exp.(-0.5 * arg1).*cos.(arg2);
#
#       K = 1;
#       for d=1:D
#           K = K .* ( h((τ[d] .^ 2)*γ[d,:], τ[d]*μ[d,:] ) * w[d,:]' );
#       end
#       K
# end

# hyp = [3,3,3,4,4,4,2.095, 2.095, 2.095]
#
# γ = 3*0.0001
#
# fn = kernel_function(SMP(hyp, 3); z=nothing)
# fn2 = kernel_function(SquaredExponentialI2(0.1, 3.0, euclidean_distance))
#
# fn([1,2,3], [5,7,8])
# fn2([1,2,3], [5,7,8])
#
# a = [0.03, 0.04]
# b = [0.0, 0.0]
# hyp = [1,2,0.1,0.4,2,1]
# fn = kernel_function(SMP(hyp, 2); z=nothing)
# fn(a,b)
#
#
# SMP(zeros(18), [1,2,3],[4,5,6])
# SMP(5*ones(18), [1,2,3],[4,5,6])
# SMP(hyp, [1,2,3],[4,5,7])
#
#
# a= [1,2,3]
# b= [7,6,5]
#
# τ = (a .- b) * 2*pi
#
# D = length(a);
# Q = Int(length(hyp)/(3*D))
#
# println(Q)
# w = reshape(  hyp[                      1:D*Q],D,Q);   # mixture weights
# μ = reshape(  hyp[D .* Q .+           (1:D*Q)],D,Q);   # spectral means
# γ = reshape(2*hyp[D .* Q .+ D .* Q .+ (1:D*Q)],D,Q);   # spectral variances
#
# w .^= 2 # square mixture weights
#
# h(arg1, arg2) = exp.(-0.5 * arg1).*cos.(arg2);
#
# K = 1
# for d=1:D
#     println(h((τ[d] .^ 2)*γ[d,:], τ[d]*μ[d,:] ))
#     println(w[d,:]')
#     K .*= ( w[d,:]' * h((τ[d] .^ 2)*γ[d,:], τ[d]*μ[d,:] ));
# end
# K
