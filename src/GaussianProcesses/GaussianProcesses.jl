"""
    GaussianProcesses

Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""
module GaussianProcesses

export
    # Gaussian process kernels
    Kernel, SquaredExponentialI, RationalQuadraticI, Matern12I, Matern32I, Matern52I, kernel_function,

    # Distance functions
    euclidean_distance, derivative_distance, antiderivative_distance,

    # Gaussian process functions
    model_output, uncertainty, compute_kernel_matrix, mean_log_marginal_loss, GPmodel, gp_model, get_kernel

using ClimateParameterizations.DataWrangling
using Flux

include("kernels.jl")
include("distances.jl")
include("gaussian_process.jl")

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])

predict(𝒱, model) = (cat((𝒱.unscale_fn(model(𝒱.training_data[i][1])) for i in 1:length(𝒱.training_data))..., dims=2), 𝒱.coarse)

function gp_model(𝒱; logγ_range=-2.0:0.1:2.0, kernel=nothing)
    function m(𝒱, kernel)
        𝒢 = GPmodel(𝒱; kernel=kernel)
        f(x) = model_output(x, GPmodel(𝒱; kernel=kernel))
        return f
    end

    if isnothing(kernel)
        best_kernel = nothing
        best_mse = Inf

        for k=1:4, logγ=logγ_range
            kernel = get_kernel(k, logγ, 0.0, euclidean_distance)
            model = m(𝒱, kernel)
            error = mse(predict(𝒱, model))

            if error < best_mse
                best_kernel = kernel
            end
        end
        return m(𝒱, best_kernel)
    else
        return m(𝒱, kernel)
    end
end

"""
    get_kernel(kernel_id::Int64, logγ, logσ, distance; logα=0.0)

# Description
Returns a Kernel object with the specified parameters.

# Arguments
- `kernel_id`: (Int64). Identification number for the kernel type (see kernel options)
- `logγ`: (Float64). Log(length scale) parameter.
- `logσ`: (Float64). Log(signal variance) parameter.
- `distance`: (Function). Distance function to use in the kernel.

# Keyword Arguments
- `logα`: (Float64). Log(α) parameter if kernel_id = 5.

# kernel options
    1   =>  "Squared exponential kernel:  k(x,x') = σ * exp( - d(x,x')² / 2γ² )",
    2   =>  "Matérn with ʋ=1/2:           k(x,x') = σ * exp( - d(x,x') / γ )",
    3   =>  "Matérn with ʋ=3/2:           k(x,x') = σ * (1+c) * exp(-√(3)*d(x,x'))/γ)",
    4   =>  "Matérn with ʋ=5/2:           k(x,x') = σ * ( 1 + √(5)*d(x,x'))/γ + 5*d(x,x')²/(3*γ^2) ) * exp(-√(5)*d(x,x'))/γ)",
    5   =>  "Rational quadratic kernel:   k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
"""
function get_kernel(kernel_id::Int64, args...)
    function g(x)
        if x isa Number
            return 10^x
        end
        return x
    end

    args = g.(args)

    kernel_id == 1 && return SquaredExponentialI(args...)
    kernel_id == 2 && return Matern12I(args...)
    kernel_id == 3 && return Matern32I(args...)
    kernel_id == 4 && return Matern52I(args...)
    kernel_id == 5 && return RationalQuadraticI(args...)
    kernel_id == 6 && return SpectralMixtureProductI(args...)

    throw(error("Invalid kernel_id $kernel_id"))
end

end #module
