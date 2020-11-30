function mean_square_error(prediction, target)
    total_error = 0.0
    gpr_prediction = predict(ℳ, 𝒟; postprocessed=true)

    # n = 𝒟.Nt-1
    n = length(gpr_prediction)
    for i in 1:n
        exact    = 𝒟.vavg[i]
        predi    = gpr_prediction[i]
        total_error += euclidean_distance(exact, predi) # euclidean distance
    end

    return total_error / n
end
export mean_square_error
