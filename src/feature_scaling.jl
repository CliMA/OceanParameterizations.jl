abstract type AbstractFeatureScaling end

#####
##### Zero-mean unit-variance feature scaling
#####

struct ZeroMeanUnitVarianceScaling{T} <: AbstractFeatureScaling
    μ :: T
    σ :: T
end

"""
    ZeroMeanUnitVarianceScaling(data)

Returns a feature scaler for `data` with zero mean and unit variance.
"""
function ZeroMeanUnitVarianceScaling(data)
    μ, σ = mean(data), std(data)
    return ZeroMeanUnitVarianceScaling(μ, σ)
end

scale(x, s::ZeroMeanUnitVarianceScaling) = (x - s.μ) / s.σ
unscale(y, s::ZeroMeanUnitVarianceScaling) = s.σ * y + s.μ

#####
##### Min-max feature scaling
#####

struct MinMaxScaling{T} <: AbstractFeatureScaling
           a :: T
           b :: T
    data_min :: T
    data_max :: T
end

"""
    MinMaxScaling(data; a=0, b=1)

Returns a feature scaler for `data` with minimum `a` and `maximum `b`.
"""
function MinMaxScaling(data; a=0, b=1)
    data_min, data_max = extrema(data)
    return MinMaxScaling{typeof(data_min)}(a, b, data_min, data_max)
end

scale(x, s::MinMaxScaling) = s.a + (x - s.data_min) * (s.b - s.a) / (s.data_max - s.data_min)
unscale(y, s::MinMaxScaling) = s.data_min + (y - s.a) * (s.data_max - s.data_min) / (s.b - s.a)

#####
##### Convinience functions
#####

(s::AbstractFeatureScaling)(x) = scale(x, s)
Base.inv(s::AbstractFeatureScaling) = y -> unscale(y, s)
