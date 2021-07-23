using Statistics: mean
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field, location
using Oceananigans.OutputReaders: FieldTimeSeries, FieldDataset

"""
    coarse_grain(Φ, n, ::Center)

Average or coarse grain a `Center`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points and `n` needs to evenly divide `length(Φ)`.
"""
function coarse_grain(Φ, n, ::Center)
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

"""
    coarse_grain(Φ, n, ::Face)

Average or coarse grain a `Face`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points. The values at the left and right endpoints of `Φ` will be preserved in the output.
"""
function coarse_grain(Φ, n, ::Face)
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = (N-2) / (n-2)

    Φ̅[1] = Φ[1]
    Φ̅[n] = Φ[N]

    if isinteger(Δ)
        Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Center())
    else
        for i in 2:n-1
            i1 = 2 + (i-2)*Δ
            i2 = 2 + (i-1)*Δ

            # Like modf but with ::Int integer part.
            f1, i1 = rem(i1, 1), trunc(Int, i1)
            f2, i2 = rem(i2, 1), trunc(Int, i2)

            left_contrib = (1 - f1) * Φ[i1]
            right_contrib = f2 * Φ[i2]
            middle_contrib = sum(Φ[i1+1:i2-1])

            Φ̅[i] = (left_contrib + middle_contrib + right_contrib) / Δ
        end
    end

    return Φ̅
end

@deprecate coarse_grain(Φ, n, loc::Type{Center}) coarse_grain(Φ, n, loc::Center)
@deprecate coarse_grain(Φ, n, loc::Type{Face}) coarse_grain(Φ, n, loc::Face)

"""
    coarse_grain(field::Field, new_grid; dims=3)

Coarse grain a `field` onto a `new_grid` along `dims`. Returns a new `Field`.
"""
function coarse_grain(field::Field{X, Y, Center}, new_grid; dims=3) where {X, Y}

    # TODO: Generalize to x and y.
    @assert dims == 3
    @assert new_grid.Nx == 1
    @assert new_grid.Ny == 1

    # TODO: Generalize `coarse_grain` to non-integer ratios.
    r = field.grid.Nz / new_grid.Nz
    @assert isinteger(r)
    r = Int(r)

    coarse_field = Field(location(field)..., field.architecture, new_grid, field.boundary_conditions)

    coarse_data = zeros(size(coarse_field))
    field_interior = interior(field)[1, 1, :]
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz, Center())

    set!(coarse_field, coarse_data)

    return coarse_field
end

function coarse_grain(field::Field{X, Y, Face}, new_grid; dims=3) where {X, Y}

    # TODO: Generalize to x and y.
    @assert dims == 3
    @assert new_grid.Nx == 1
    @assert new_grid.Ny == 1

    r = field.grid.Nz / new_grid.Nz

    coarse_field = Field(location(field)..., field.architecture, new_grid, field.boundary_conditions)

    coarse_data = zeros(size(location(coarse_field), new_grid))
    field_interior = interior(field)[1, 1, :]
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz+1, Face())

    set!(coarse_field, coarse_data)

    return coarse_field
end

function coarse_grain(fts::FieldTimeSeries, new_grid)
    fts_new = FieldTimeSeries(new_grid, location(fts), fts.times; ArrayType=Array{Float32})

    Nt = size(fts, 4)
    for n in 1:Nt
        fts_new.data[:, :, :, n] .= coarse_grain(fts[n], new_grid).data
    end

    return fts_new
end

function coarse_grain(fds::FieldDataset, new_grid)
    coarse_fields = Dict{String, FieldTimeSeries}(
        name => coarse_grain(fts, new_grid)
        for (name, fts) in fds.fields
    )
    return FieldDataset(coarse_fields, fds.metadata, fds.filepath)
end
