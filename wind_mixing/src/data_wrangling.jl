using Oceananigans
using Oceananigans.OutputReaders
using JLD2

using Statistics: mean
using Oceananigans.Fields: Field, location
using Oceananigans.OutputReaders
using OceanParameterizations

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
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz, Center)

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
    coarse_data[1, 1, :] .= OceanParameterizations.coarse_grain(field_interior, new_grid.Nz+1, Face)

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

function add_scaled_data!(ds::FieldDataset, scalings)
    apply_surface_fluxes!(ds)
    ds.fields["u*"] = deepcopy(ds["u"])
    ds.fields["v*"] = deepcopy(ds["v"])
    ds.fields["T*"] = deepcopy(ds["T"])
    ds.fields["wu*"] = deepcopy(ds["wu"])
    ds.fields["wv*"] = deepcopy(ds["wv"])
    ds.fields["wT*"] = deepcopy(ds["wT"])

    interior(ds["u*"]) .= scalings.u.(interior(ds["u"]))
    interior(ds["v*"]) .= scalings.v.(interior(ds["v"]))
    interior(ds["T*"]) .= scalings.T.(interior(ds["T"]))

    interior(ds["wu*"]) .= scalings.uw.(interior(ds["wu"]))
    interior(ds["wv*"]) .= scalings.vw.(interior(ds["wv"]))
    interior(ds["wT*"]) .= scalings.wT.(interior(ds["wT"]))
    nothing
end

function add_scaled_gradients!(ds::FieldDataset)
    @assert haskey(ds.fields, "u*") && haskey(ds.fields, "T*") && haskey(ds.fields, "T*")
    Nz = ds["u"].grid.Nz
    D_face = Float32.(Dᶠ(Nz, 1 / Nz))

    ds.fields["∂u∂z*"] = deepcopy(ds["wu*"])
    ds.fields["∂v∂z*"] = deepcopy(ds["wu*"])
    ds.fields["∂T∂z*"] = deepcopy(ds["wu*"])

    u = interior(ds["u*"])
    v = interior(ds["v*"])
    T = interior(ds["T*"])

    @inline function calculate_gradients!(input, output)
        for i in 1:size(output, 1), j in 1:size(output, 2), t in 1:size(output, 4)
            col = @view output[i, j, :, t]
            col .= D_face * input[i, j, :, t]
        end
    end

    calculate_gradients!(u, interior(ds["∂u∂z*"]))
    calculate_gradients!(v, interior(ds["∂v∂z*"]))
    calculate_gradients!(T, interior(ds["∂T∂z*"]))

    nothing
end

struct TrainingScaling{𝒮}
    u::𝒮
    v::𝒮
    T::𝒮

    uw::𝒮
    vw::𝒮
    wT::𝒮
end

function TrainingScaling(datasets_coarse, scaling)
    !isa(datasets_coarse, Array) && (datasets_coarse = [datasets_coarse])

    us_unscaled = vcat([interior(dataset["u"])[1,1,:,:] for dataset in datasets_coarse]...)
    vs_unscaled = vcat([interior(dataset["v"])[1,1,:,:] for dataset in datasets_coarse]...)
    Ts_unscaled = vcat([interior(dataset["T"])[1,1,:,:] for dataset in datasets_coarse]...)

    uws_unscaled = vcat([interior(dataset["wu"])[1,1,:,:] for dataset in datasets_coarse]...)
    vws_unscaled = vcat([interior(dataset["wv"])[1,1,:,:] for dataset in datasets_coarse]...)
    wTs_unscaled = vcat([interior(dataset["wT"])[1,1,:,:] for dataset in datasets_coarse]...)

    u_scaling = scaling(us_unscaled)
    v_scaling = scaling(vs_unscaled)
    T_scaling = scaling(Ts_unscaled)

    uw_scaling = scaling(uws_unscaled)
    vw_scaling = scaling(vws_unscaled)
    wT_scaling = scaling(wTs_unscaled)

    TrainingScaling(u_scaling, v_scaling, T_scaling, uw_scaling, vw_scaling, wT_scaling)
end

struct TrainingDatasets{𝒮, 𝒯}
    data::𝒮
    scalings::𝒯
end

function TrainingDatasets(datasets::Vector{FieldDataset{Dict{String, FieldTimeSeries}, Dict{String, Any}, String}}; 
                            Nz_coarse=32, scaling=ZeroMeanUnitVarianceScaling)
    for ds in datasets
        apply_surface_fluxes!(ds)
    end

    les_grid = datasets[1]["T"].grid

    topo = (Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded)
    domain = (les_grid.zF[1], les_grid.zF[les_grid.Nz+1])
    coarse_grid = RegularRectilinearGrid(Float32, topology=topo, size=Nz_coarse, z=domain)

    datasets_coarse = [coarse_grain(ds, coarse_grid) for ds in datasets]

    for ds in datasets_coarse
        add_scaled_data!(ds, scalings)
        add_scaled_gradients!(ds)
    end

    scalings = TrainingScaling(datasets_coarse, scaling)

    TrainingDatasets(datasets_coarse, scalings)
end

function TrainingDatasets(dirnames; Nz_coarse=32, scaling=ZeroMeanUnitVarianceScaling)
    dirnames isa String && (dirnames = [dirnames])
    
    FILE_PATHs = [joinpath(pwd(), "Data", dirname, "instantaneous_statistics_with_halos.jld2") for dirname in dirnames]

    datasets = [FieldDataset(PATH, ArrayType=Array{Float32}, metadata_paths=["parameters"]) for PATH in FILE_PATHs]

    for ds in datasets
        apply_surface_fluxes!(ds)
        for (key, value) in ds.metadata
            isa(value, Real) && !isa(value, Integer) && (ds.metadata[key] = Float32(value))
        end
    end

    fine_grid = datasets[1]["T"].grid

    domain = (fine_grid.zF[1], fine_grid.Nz+1)
    coarse_grid = RegularRectilinearGrid(Float32, topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded), size=Nz_coarse, z=domain)

    datasets_coarse = [coarse_grain(ds, coarse_grid) for ds in datasets]

    scalings = TrainingScaling(datasets_coarse, scaling)

    for ds in datasets_coarse
        add_scaled_data!(ds, scalings)
        add_scaled_gradients!(ds)
    end

    TrainingDatasets(datasets_coarse, scalings)
end

function apply_surface_fluxes!(ds)
    interior(ds["wu"])[:, :, end, :] .= ds.metadata["momentum_flux"]
    interior(ds["wT"])[:, :, end, :] .= ds.metadata["temperature_flux"]
end