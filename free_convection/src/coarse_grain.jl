using DimensionalData: basetypeof
using GeoData: AbstractGeoStack

function coarse_grain(A::Array, n, ::Cell)
    N = length(A)
    Δ = Int(N / n)
    Ā = similar(A, n)
    for i in 1:n
        Ā[i] = mean(A[Δ*(i-1)+1:Δ*i])
    end
    return Ā
end

function coarse_grain(A::Array, n, ::Face)
    N = length(A)
    gap = (N-1)/n

    Ā = similar(A, n+1)

    # Left and right faces must match
    Ā[1] = A[1]
    Ā[end] = A[end]

    for i in 2:n
        Ā[i] = 1 + (i-1) * gap
    end

    for i=2:n
        Ā[i] = (floor(Ā[i])+1 - Ā[i]) * A[Int(floor(Ā[i]))] + (Ā[i] - floor(Ā[i])) * A[Int(floor(Ā[i]))+1]
    end

    return Ā
end

function coarse_grain(d::Dimension, n, loc)
    dim_base_type = basetypeof(d)
    d̄ = coarse_grain(val(d), n, loc)
    return dim_base_type(d̄, mode=mode(d), metadata=metadata(d))
end

function coarse_grain(A::GeoArray, n)
    loc = hasdim(A, zC) ? Cell() : Face()

    N = size(A, ZDim)
    T = size(A, Ti)

    Ā = zeros(eltype(A), loc isa Face ? n+1 : n, T)
    for j in 1:T
        Ā[:, j] .= coarse_grain(data(A[Ti=j]), n, loc)
    end

    z = dims(A, ZDim)
    coarse_grained_dims = (coarse_grain(z, n, loc), otherdims(A, z)...)
    return GeoArray(Ā, dims=coarse_grained_dims, name=GeoData.name(A), refdims=refdims(A), metadata=metadata(A), missingval=missingval(A))
end
