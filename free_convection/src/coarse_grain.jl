using Statistics: mean
using Oceananigans.Fields: Field, location
using Oceananigans.OutputReaders: FieldTimeSeries, FieldDataset

import OceanParameterizations: coarse_grain

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
