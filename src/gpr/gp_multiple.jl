"""
Adapted from sandreza/Learning/sandbox/gaussian_process.jl
https://github.com/sandreza/Learning/blob/master/sandbox/gaussian_process.jl
Changed handling of kernel functions; changed some variable names;
added log marginal likelihood function.
"""

using LinearAlgebra

"""
GP_multiple
# Description
- data structure for GPR computations where each gridpoint in the prediction has a different predictor
# Data Structure and Description
    GPs, Array of GP objects
    kernel, Kernel object
    x_train
"""
struct GP_multiple
    GPs::Array{GP}
    kernel::Kernel
    x_train
end

function stencil_range(D, stencil_size, i)
    k = Int(floor(stencil_size/2))
    if i-k < 1
        return 1:stencil_size
    elseif i-k+stencil_size-1 > D
        return D-stencil_size+1:D
    else
        start = i-k
        start:start+stencil_size-1
    end
end

stencil(stencil_range, data) = [x[stencil_range] for x in data]

function model(𝒟::ProfileData; kernel::Kernel = Kernel(), stencil_size=)
    # create instance of GP using data from ProfileData object
    stencil_ranges = [stencil_range(D,stencil_size,i) for i=1:D]
    GPs = [model(stencil(𝒟.x_train,r), stencil(𝒟.y_train,r), kernel, 𝒟.zavg[r]) for range in stencil_ranges]
    return GP_multiple(GPs, kernel, 𝒟.x_train);
end

"""
prediction(x, 𝒢::GP)
# Description
- Given state x, GP 𝒢, returns the mean GP prediction
# Arguments
- `x`: single scaled state
- `𝒢`: GP object with which to make the prediction
# Return
- `y`: scaled prediction
"""
function model_output(x, 𝒢::GP_multiple)
    return [model_output(x[i],𝒢.GPs[i]) for i in 1:length(x)]
end

"""
uncertainty(x, 𝒢::GP)
# Description
- Given state x and GP 𝒢, output the variance at a point
# Arguments
- `x`: state
# Return
- `var`: variance
"""
function uncertainty(x, 𝒢::GP)
    tmpv = zeros(size(𝒢.x_train)[1])
    for i in eachindex(𝒢.x_train)
        tmpv[i] = 𝒢.kernel(x, 𝒢.x_train[i])
    end
    # no ldiv for suitesparse
    tmpv2 = 𝒢.CK \ tmpv
    var = k(x, x) .- tmpv'*tmpv2  # var(f*) = k(x*,x*) - tmpv'*tmpv2
    return var
end

"""
compute_kernel_matrix(kernel, x)
# Description
- Computes the kernel matrix for GPR
# Arguments
- `k` : (Kernel) kernel function k(a,b).
- `x` : (array of predictors). x[1] is a vector
# Return
- `sK`: (symmetric matrix). A symmetric matrix with entries sK[i,j] = k(x[i], x[j]). This is only meaningful if k(x,y) = k(y,x) (it should)
"""
function compute_kernel_matrix(k, x)

    K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]

    if typeof(K[1,1]) <: Number
        sK = Symmetric(K)
    else
        sK = K
    end
    return sK
end

"""
mean_log_marginal_loss(y_train, 𝒢::GP; add_constant=false)
# Description
Computes log marginal loss for each element in the output and averages the results.
Assumes noise-free observations.

log(p(y|X)) = -(1/2) * (y'*α + 2*sum(Diagonal(CK)) + n*log(2*pi))
where n is the number of training points and

# Arguments
- `y_train`: (Array). training outputs (prediction), must have the same number as x_train
- `𝒢`: (GP).
# Keyword Arguments
- `add_constant`: (bool). whether to give the exact value of the loss or leave out an added constant for efficiency.

"""
function mean_log_marginal_loss(y_train, 𝒢::GP; add_constant=false)
    n = length(𝒢.x_train)
    D = length(𝒢.x_train[1])

    ys = hcat(y_train...)' # n x D

    if add_constant
        c = sum([log(𝒢.CK[i,i]) for i in 1:n]) + 0.5*n*log(2*pi)
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i] + c
        end
    else
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i]
        end
    end

    return total_loss / D
end
