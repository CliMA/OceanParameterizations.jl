"""
    FreeConvectionTrainingDataInput{Θ, B, T}

A container for holding free convection training data inputs.
"""
struct FreeConvectionTrainingDataInput{Θ, B, T}
    temperature :: Θ
    bottom_flux :: B
       top_flux :: T
end

rescale(old, T_scaling, wT_scaling) =
    FreeConvectionTrainingDataInput(T_scaling.(old.temperature), wT_scaling(old.bottom_flux), wT_scaling(old.top_flux))

function wrangle_input_training_data(datasets; use_missing_fluxes)
    data = []

    for (id, ds) in datasets
        T = ds["T"]

        wT = use_missing_fluxes ? ds["wT_missing"] : ds["wT"]

        Nz = size(T, 3)
        Nt = size(T, 4)

        data_id = [
            FreeConvectionTrainingDataInput(
                interior(T)[1, 1, :, n],
                interior(wT)[1, 1, 1, n],
                interior(wT)[1, 1, Nz+1, n]
            )
            for n in 1:Nt
        ]

        push!(data, data_id)
    end

    return cat(data..., dims=1)
end

function wrangle_output_training_data(datasets; use_missing_fluxes)

    data = []

    for (id, ds) in datasets
        wT = use_missing_fluxes ? ds["wT_missing"] : ds["wT"]
        push!(data, interior(wT)[1, 1, :, :])
    end

    return cat(data..., dims=2)
end
