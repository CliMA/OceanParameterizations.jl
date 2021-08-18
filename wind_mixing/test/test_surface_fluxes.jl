using WindMixing: ConstantMomentumFlux
using WindMixing: ConstantTemperatureFlux
using WindMixing: SinusoidalMomentumFlux
using WindMixing: SinusoidalTemperatureFlux

@testset "Surface Fluxes" begin
    Qᵘ = -5e-4
    Tᵘ = 24.
    ϕᵘ = π / 2
    Cᵘ = -1e-4

    Qᵇ = -5e-4
    Tᵇ = 24.
    ϕᵇ = π / 2
    Cᵇ = -1e-4
    α = 1.67e-4
    g = 9.81

    Qᵘ_constant = ConstantMomentumFlux(Qᵘ)
    Qᶿ_constant = ConstantTemperatureFlux(Qᵇ, α, g)

    Qᵘ_sinusoidal = SinusoidalMomentumFlux(Qᵘ, Tᵘ, ϕᵘ, Cᵘ)
    Qᶿ_sinusoidal = SinusoidalTemperatureFlux(Qᵇ, Tᵇ, ϕᵇ, Cᵇ, α, g)


    @test typeof(Qᵘ_constant(rand(), rand(), 5.)) == Float64
    @test typeof(Qᵘ_constant(rand(), rand(), 5f0)) == Float32
    @test typeof(Qᶿ_constant(rand(), rand(), 5.)) == Float64
    @test typeof(Qᶿ_constant(rand(), rand(), 5f0)) == Float32

    @test Qᵘ_constant(rand(), rand(), rand()) == Qᵘ
    @test Qᶿ_constant(rand(), rand(), rand()) == Qᵇ / (α * g)

    @test typeof(Qᵘ_sinusoidal(rand(), rand(), 5.)) == Float64
    @test typeof(Qᵘ_sinusoidal(rand(), rand(), 5f0)) == Float32
    @test typeof(Qᶿ_sinusoidal(rand(), rand(), 5.)) == Float64
    @test typeof(Qᶿ_sinusoidal(rand(), rand(), 5f0)) == Float32

    @test Qᵘ_sinusoidal(rand(), rand(), 0.) == Qᵘ * sin(ϕᵘ) + Cᵘ
    @test Qᶿ_sinusoidal(rand(), rand(), 0.) == (Qᵇ * sin(ϕᵇ) + Cᵇ) / (α  * g)

    @test Qᵘ_sinusoidal(rand(), rand(), 0.) == Qᵘ_sinusoidal(rand(), rand(), Tᵘ * 60 ^ 2)
    @test Qᶿ_sinusoidal(rand(), rand(), 0.) == Qᶿ_sinusoidal(rand(), rand(), Tᵘ * 60 ^ 2)
end
