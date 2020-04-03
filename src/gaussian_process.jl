using LinearAlgebra

"""
    GaussianProcess{K, D, P, M, C}

# Description
Data structure for typical GPR computations.

# Data Structure and Description
    kernel::K, a function
    data::D, an array of vectors
    predictor::P, an array
    K::M, matrix or sparse matrix
    CK::C, cholesky factorization of K
"""
struct GaussianProcess{K, D, P, M, C}
       kernel :: K
         data :: D
    predictor :: P
            K :: M
           CK :: C
end

"""
GaussianProcess(x, y, kernel; hyperparameters = [], sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

# Description
Constructs the posterior distribution for a GP. In other words this does the 'training' automagically.

# Arguments
- 'x': (array). predictor, must be an array of states
- 'y': (array). prediction, must have the same number as x
- 'kernel': (function). maps predictor x predictor to real numbers

# Keyword Arguments
- 'hyperparameters': (array). default = []. hyperparameters that enter into the kernel
- 'sparsity_threshold': (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it
- 'robust': (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.
- 'entry_threshold': (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
# Return
- 'GP Object': (GP).

"""
function GaussianProcess(x, y, kernel; hyperparameters=[], sparsity_threshold=0.0,
                         robust=true, entry_threshold=√(eps(1.0)))
    K = compute_kernel_matrix(kernel, x)

    # Get the maximum entry for scaling and sparsity checking.
    K_max = maximum(K)

    # Make Cholesky factorization work by adding a small amount to the diagonal.
    if robust
        K += K_max * entry_threshold * I
    end

    # Check sparsity, should make this a seperate Module.
    bools = K .> entry_threshold * K_max
    sparsity = sum(bools) / length(bools)

    if sparsity < sparsity_threshold
        sparse_K = similar(K) .* 0
        sparse_K[bools] = sK[bools]
        K = sparse(Symmetric(sparse_K))
        CK = cholesky(K)
    else
        CK = cholesky(K)
    end

    y = hcat(y...)'
    predictor = CK \ y

    return GaussianProcess(kernel, x, predictor, K, CK)
end

"""
predict(gp, x)

# Description
- Given state x and GP, make a prediction

# Arguments
- 'x': state

# Return
- 'y': prediction
"""
predict(gp, x) = gp.predictor' * gp.kernel.(x, gp.data)

"""
uncertainty(gp, x)

# Description
- Given state x and GP, output the variance at a point

# Arguments
- 'x': state

# Return
- 'var': variance
"""
function uncertainty(gp, x)
    tmpv = zeros(size(gp.data)[1])
    for i in eachindex(gp.data)
        tmpv[i] = gp.kernel(x, 𝒢.data[i])
    end

    # no ldiv for suitesparse
    tmpv2 = gp.CK \ tmpv
    var = k(x, x) - tmpv' * tmpv2

    return var
end

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
        K = [k(x[i], x[j], hyperparameters=hyperparameters) for i in eachindex(x), j in eachindex(x)]
    end

    if typeof(K[1,1]) <: Number
        sK = Symmetric(K)
    else
        sK = K
    end

    return sK
end
