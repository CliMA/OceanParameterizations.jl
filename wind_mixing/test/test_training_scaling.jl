using WindMixing: calculate_training_scalings

@testset "Training Scaling" begin
    losses = (u=rand(Float32), v=rand(Float32), T=rand(Float32), ∂u∂z=rand(Float32), ∂v∂z=rand(Float32), ∂T∂z=rand(Float32))
    fractions = (T=rand(Float32), ∂T∂z=rand(Float32), profile=rand(Float32))
    
    scalings = calculate_training_scalings(losses, fractions)

    u = scalings.u * losses.u
    v = scalings.v * losses.v
    T = scalings.T * losses.T

    ∂u∂z = scalings.∂u∂z * losses.∂u∂z
    ∂v∂z = scalings.∂v∂z * losses.∂v∂z
    ∂T∂z = scalings.∂T∂z * losses.∂T∂z

    @test T / (u + v) ≈ fractions.T / (1 - fractions.T)
    @test ∂T∂z / (∂u∂z + ∂v∂z) ≈ fractions.∂T∂z / (1 - fractions.∂T∂z)
    @test (u + v + T) / (∂u∂z + ∂v∂z + ∂T∂z) ≈ fractions.profile / (1 - fractions.profile)
end