using Oceananigans: Flat, Bounded, RegularRectilinearGrid

ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/neural_free_convection_training_data"

SIMULATION_IDS = 1:21

LESBRARY_DATA_DEPS = [
    DataDep("free_convection_$id",
            "LESbrary.jl free convection simulation $id",
            joinpath(ENGAGING_LESBRARY_DIR, "free_convection_$id", "instantaneous_statistics_with_halos.jld2")
    ) for id in SIMULATION_IDS
]

function validate_simulation_ids(ids_train, ids_test)
    @info "Simulation IDs: $(collect(SIMULATION_IDS))"
    @info "Training simulations: $(collect(ids_train))"
    @info "Testing simulations: $(collect(ids_test))"

    ids_not_used = setdiff(SIMULATION_IDS, ids_train, ids_test)
    if !isempty(ids_not_used)
        @warn "Simulations not used: $(collect(ids_not_used))"
    end

    ids_intersection = intersect(ids_train, ids_test)
    if !isempty(ids_intersection)
        @warn "Simulations used for both training and testing: $(collect(ids_intersection))"
    end

    return nothing
end

function load_data(ids_train, ids_test, Nz; convective_adjustment_K)

    @info "Constructing FieldTimeSeries..."

    grid_1d = RegularRectilinearGrid(topology=(Flat, Flat, Bounded), size=128, extent=256, halo=3)

    datasets = Dict{Int, FieldDataset}(
        id => FieldDataset(@datadep_str "free_convection_$id/instantaneous_statistics_with_halos.jld2"; grid=grid_1d, ArrayType=Array{Float32}, metadata_paths=["parameters"])
        for id in SIMULATION_IDS
    )

    @info "Injecting surface fluxes..."

    for ds in values(datasets)
        add_surface_fluxes!(ds)
    end

    @info "Diagnosing convective adjustment fluxes..."

    for ds in values(datasets)
        add_convective_adjustment_flux!(ds, convective_adjustment_K)
    end

    @info "Coarsening grid..."

    les_grid = datasets[1]["T"].grid

    topo = (Flat, Flat, Bounded)
    domain = (les_grid.zF[1], les_grid.zF[les_grid.Nz+1])
    coarse_grid = RegularRectilinearGrid(Float32, topology=topo, size=Nz, z=domain)

    @info "Coarse graining data..."

    coarse_datasets = Dict{Int, FieldDataset}(
        id => coarse_grain(ds, coarse_grid)
        for (id, ds) in datasets
    )

    @info "Partitioning data into training and testing datasets..."

    training_datasets = Dict{Int, FieldDataset}(id => datasets[id] for id in ids_train)
    testing_datasets = Dict{Int, FieldDataset}(id => datasets[id] for id in ids_test)

    coarse_training_datasets = Dict{Int, FieldDataset}(id => coarse_datasets[id] for id in ids_train)
    coarse_testing_datasets = Dict{Int, FieldDataset}(id => coarse_datasets[id] for id in ids_test)

    return (; datasets, coarse_datasets, training_datasets, testing_datasets, coarse_training_datasets, coarse_testing_datasets)
end
