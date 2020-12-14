const ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary/free_convection_training_data/"

const LESBRARY_DATA_DEPS = (
    DataDep("free_convection_Qb1e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 1×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb1.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb1e-8", "statistics.nc"),
            "f935dbc46281c478141053673145b32551c1656921992fd81e25a467cea106ea"),
    DataDep("free_convection_Qb2e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 2×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb2.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb2e-8", "statistics.nc"),
            "9bad22e7ceb7f5bb8a562d222869b37ed331771c451af1a03fcafb23360e51ee"),
    DataDep("free_convection_Qb3e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 3×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb3.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb3e-8", "statistics.nc"),
            "3c19a33357c7822c886ad1aac4d811d5818e03067956b6124f8ec24921919c57"),
    DataDep("free_convection_Qb4e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 4×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb4.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb4e-8", "statistics.nc"),
            "2a7813826a5b1109983b7761971a584b0f78f49fd30fadb3a444c87e252a0bbd"),
    DataDep("free_convection_Qb5e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 5×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb5e-8", "statistics.nc"),
            "db31f7dba27c0d6f4b33cf371110dbf9c7ffc4c8ae4f22abac6920f6ca10c56c"),
    DataDep("free_convection_Qb6e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 6×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb6.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb6e-8", "statistics.nc"),
            "1b3b592e64b7cebc417982a309a10193535f96ac663d62645372df87afc5d789")
)
