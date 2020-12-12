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

function input_training_data(training_datasets)
    data = []

    for (id, ds) in training_datasets
        Nz, Nt = size(ds[:T])

        subdata = [FreeConvectionTrainingDataInput(
                    ds[:T][Ti=n].data,
                    ds[:wT][zF=1, Ti=n],
                    ds[:wT][zF=Nz+1, Ti=n])
                   for n in 1:Nt]

        push!(data, subdata)
    end

    return cat(data..., dims=1)
end

function output_training_data(training_datasets)
    data = []
    for (id, ds) in training_datasets
        push!(data, ds[:wT].data)
    end
    return cat(data..., dims=2)
end
