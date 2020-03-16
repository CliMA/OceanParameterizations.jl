using LinearAlgebra

"""
GP
# Description
- data structure for typical GPR computations
# Data Structure and Description
    kernel::ℱ, a function
    data::𝒮 , an array of vectors
    predictor::𝒮2 , an array
    K::𝒰 , matrix or sparse matrix
    CK::𝒱, cholesky factorization of K
"""
struct GP{ℱ, 𝒮, 𝒮2, 𝒰, 𝒱}
    kernel::ℱ
    data::𝒮
    predictor::𝒮2
    K::𝒰
    CK::𝒱
end


"""
construct_gpr(x_data, y_data, kernel; hyperparameters = [], sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

# Description
Constructs the posterior distribution for a GP. In other words this does the 'training' automagically.

# Arguments
- 'x_data': (array). predictor, must be an array of states
- 'y_data': (array). prediction, must have the same number as x_data
- 'kernel': (function). maps predictor x predictor to real numbers

# Keyword Arguments
- 'hyperparameters': (array). default = []. hyperparameters that enter into the kernel
- 'sparsity_threshold': (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it
- 'robust': (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.
- 'entry_threshold': (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
# Return
- 'GP Object': (GP).

"""
function construct_gpr(x_data, y_data, kernel; hyperparameters = [], sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))
    K = compute_kernel_matrix(kernel, x_data)
    # get the maximum entry for scaling and sparsity checking
    mK = maximum(K)

    # make Cholesky factorization work by adding a small amount to the diagonal
    if robust
        K += mK*sqrt(eps(1.0))*I
    end

    # check sparsity, should make this a seperate Module
    bools = K .> entry_threshold * mK
    sparsity = sum(bools) / length(bools)
    if sparsity < sparsity_threshold
        sparse_K = similar(K) .* 0
        sparse_K[bools] = sK[bools]
        K = sparse(Symmetric(sparse_K))
        CK = cholesky(K)
    else
        CK = cholesky(K)
    end

    y = hcat(y_data...)'
    predictor = CK \ y

    return GP(kernel, x_data, predictor, K, CK)
end

"""
prediction(x, 𝒢::GP)

# Description
- Given state x and GP 𝒢, make a prediction

# Arguments
- 'x': state

# Return
- 'y': prediction
"""
function prediction(x, 𝒢::GP)
    y =  𝒢.predictor' * 𝒢.kernel.(x, 𝒢.data)
    return y
end

"""
uncertainty(x, 𝒢::GP)

# Description
- Given state x and GP 𝒢, output the variance at a point

# Arguments
- 'x': state

# Return
- 'var': variance
"""
function uncertainty(x, 𝒢::GP)
    tmpv = zeros(size(𝒢.data)[1])
    for i in eachindex(𝒢.data)
        tmpv[i] = 𝒢.kernel(x, 𝒢.data[i])
    end
    # no ldiv for suitesparse
    tmpv2 = 𝒢.CK \ tmpv
    var = k(x, x) - tmpv'*tmpv2
    return var
end

###
"""
compute_kernel_matrix(k, x)

# Description
- Computes the kernel matrix for GPR

# Arguments
- k : (function) the kernel. Takes in two arguments and produce a real number
- x : (array of predictors). x[1] is a vector

# Return
- sK: (symmetric matrix). A symmetric matrix with entries sK[i,j] = k(x[i], x[j]). This is only meaningful if k(x,y) = k(y,x) (it should)
"""
function compute_kernel_matrix(k, x; hyperparameters = [])
    if isempty(hyperparameters)
        K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]
    else
        K = [k(x[i], x[j], hyperparameters = hyperparameters) for i in eachindex(x), j in eachindex(x)]
    end

    if typeof(K[1,1]) <: Number
        sK = Symmetric(K)
    else
        sK = K
    end
    return sK
end


"""
gaussian_kernel(x,y; γ = 1.0, σ = 1.0)

# Description
- Outputs a Gaussian kernel with hyperparameter γ

# Arguments
- x: first coordinate
- y: second coordinate

# Keyword Arguments
-The first is γ, the second is σ where, k(x,y) = σ * exp(- γ * d(x,y))
- γ = 1.0: (scalar). hyperparameter in the Gaussian Kernel.
- σ = 1.0; (scalar). hyperparameter in the Gaussian Kernel.
"""
function gaussian_kernel(x,y; γ = 1.0, σ = 1.0)
    y = σ * exp(- γ * d(x,y))
    return y
end

"""
closure_gaussian_kernel(x,y; γ = 1.0, σ = 1.0)

# Description
- Outputs a function that computes a Gaussian kernel

# Arguments
- d: distance function. d(x,y)

# Keyword Arguments
-The first is γ, the second is σ where, k(x,y) = σ * exp(- γ * d(x,y))
- γ = 1.0: (scalar). hyperparameter in the Gaussian Kernel.
- σ = 1.0; (scalar). hyperparameter in the Gaussian Kernel.
"""
function closure_guassian_closure(d; hyperparameters = [1.0, 1.0])
    function gaussian_kernel(x,y)
        y = hyperparameters[2] * exp(- hyperparameters[1] * d(x,y))
        return y
    end
    return gaussian_kernel
end
