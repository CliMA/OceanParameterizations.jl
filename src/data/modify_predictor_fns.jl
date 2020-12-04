"""Options for the `modify_predictor_fn` keyword argument when defining a `Problem`."""

function append_tke(x, time_index, state_variables)
    vcat(x, state_variables.tke_avg[time_index])
end

function partial_temp_profile(z_set)
    function f(x, time_index, 𝒟)
        x[z_set]
    end
    f
end
