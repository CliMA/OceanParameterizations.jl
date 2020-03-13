weights(f::FastDense, θ) = reshape(θ[1:(f.out*f.in)], f.out, f.in)
bias(f::FastDense, θ) = θ[(f.out*f.in+1):end]

