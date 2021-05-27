ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary/neural_free_convection_training_data"

LESBRARY_DATA_DEPS = [
    DataDep("free_convection_$id",
            "LESbrary.jl free convection simulation $id",
            joinpath(ENGAGING_LESBRARY_DIR, "free_convection_$id", "instantaneous_statistics_with_halos.jld2")
    ) for id in 1:9
]
