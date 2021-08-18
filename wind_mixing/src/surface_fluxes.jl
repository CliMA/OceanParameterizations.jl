abstract type SurfaceFlux end

abstract type SinusoidalFlux <: SurfaceFlux end
abstract type ConstantFlux <: SurfaceFlux end

struct ConstantMomentumFlux{T} <: ConstantFlux
    Q::T
end

struct ConstantTemperatureFlux{T} <: ConstantFlux
    Q::T
    α::T
    g::T
end

struct SinusoidalMomentumFlux{T} <: SinusoidalFlux
    Q::T
    T::T
    ϕ::T
    C::T
end

struct SinusoidalTemperatureFlux{T} <: SinusoidalFlux
    Q::T
    T::T
    ϕ::T
    C::T
    α::T
    g::T
end

(s::ConstantMomentumFlux)(x, y, t)::typeof(t) = s.Q

(s::ConstantTemperatureFlux)(x, y, t)::typeof(t) = s.Q / (s.α * s.g)

(s::SinusoidalMomentumFlux)(x, y, t)::typeof(t) = s.Q * sin(2π / (s.T * 60 ^ 2) * t + s.ϕ) + s.C

(s::SinusoidalTemperatureFlux)(x,y,t)::typeof(t) = (s.Q * sin(2π / (s.T * 60 ^ 2) * t + s.ϕ) + s.C) / (s.α * s.g)
