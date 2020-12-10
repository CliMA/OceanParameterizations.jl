"""
Adapted from sandreza/Learning/sandbox/gaussian_process.jl
https://github.com/sandreza/Learning/blob/master/sandbox/gaussian_process.jl
Changed handling of kernel functions, some variable names, and added log marginal likelihood function.
"""

using LinearAlgebra

"""
    GP{K, 𝒮, 𝒮2, 𝒰, 𝒱, 𝒜}

Data structure for typical Gaussian process regression computations.
"""
struct GP{K, 𝒮, 𝒮2, 𝒰, 𝒱, 𝒜}
     kernel :: K  # a Kernel object
    x_train :: 𝒮  # an array of vectors (n-length array of D-length vectors)
          α :: 𝒮2 # an array
          K :: 𝒰  # matrix or sparse matrix
         CK :: 𝒱  # cholesky factorization of K
      cache :: 𝒜
end

"""
    GPmodel(x_train, y_train, kernel, zavg; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

Constructs the posterior distribution for a Gaussian process. In other words this does the "training" automagically.

# Arguments
- `x_train`: (array). training inputs (predictors), must be an array of states.
                      length-n array of D-length vectors, where D is the length of each input n is the number of training points.
- `y_train`: (array). training outputs (prediction), must have the same number as x_train
                      length-n array of D-length vectors.
- `kernel`: (Kernel). Kernel object. See kernels.jl.
                      kernel_function(kernel)(x,x') maps predictor x predictor to real numbers.
# Keyword Arguments
- `z`: (vector). values w.r.t. which to derivate the state (default none).

- `normalize`: (bool). whether to normalize the data during preprocessing and reverse the scaling for postprocessing. Can lead to better performance.

- `hyperparameters`: (array). default = []. hyperparameters that enter into the kernel

- `sparsity_threshold`: (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it

- `robust`: (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.

- `entry_threshold`: (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
"""
function GPmodel(x_train, y_train, kernel, zavg; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

    # get k(x,x') function from kernel object
    kernel = kernel_function(kernel; z=zavg)

    # fill kernel matrix with values
    K = compute_kernel_matrix(kernel, x_train)

    # get the maximum entry for scaling and sparsity checking
    mK = maximum(K)

    # make Cholesky factorization work by adding a small amount to the diagonal
    if robust
        K += mK*sqrt(eps(1.0))*I
    end

    # check sparsity
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

    # get prediction weights FIX THIS SO THAT IT ALWAYS WORKS
    y = hcat(y_train...)'
    α = CK \ y # α = K + σ_noise*I

    # construct struct
    return GP(kernel, x_train, α', K, Array(CK), zeros(length(x_train)))
end

"""
    GPmodel(𝒱; kernel::Kernel = Kernel())

Create an instance of GP using data from ProfileData object 𝒟.

# Arguments
- 𝒱::VData, Data for training the GP
"""
function GPmodel(𝒱; kernel::Kernel = Kernel())
    x_train = [pair[1] for pair in 𝒱.training_data]
    y_train = [pair[2] for pair in 𝒱.training_data]

    # create instance of GP using data from ProfileData object
    return GPmodel(x_train, y_train, kernel, 𝒱.z);
end

"""
prediction(x, 𝒢::GP)

Given state x, GP 𝒢, returns the mean GP prediction

# Arguments
- `x`: single scaled state
- `𝒢`: GP object with which to make the prediction

# Return
- `y`: scaled prediction
"""
function model_output(x, 𝒢::GP)
    for i in 1:length(𝒢.cache)
        𝒢.cache[i] = 𝒢.kernel(x, 𝒢.x_train[i])
    end
    return 𝒢.α * 𝒢.cache
end

"""
    uncertainty(x, 𝒢::GP)

Given state x and GP 𝒢, output the variance at a point

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
    var = k(x, x) .- tmpv'*tmpv2
    return var
end

"""
    compute_kernel_matrix(kernel, x)

Computes the kernel matrix for GPR.

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

Computes log marginal loss for each element in the output and averages the results.

Assumes noise-free observations

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
