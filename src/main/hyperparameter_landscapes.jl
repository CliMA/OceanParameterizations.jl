using Plots
using StaticArrays

function train_validate_test(𝒟_train, 𝒟_validate, 𝒟_test, problem; log_γs=-1.0:0.1:1.0, distances=[euclidean_distance, derivative_distance, antiderivative_distance],descriptor="")
    # Train GP on the filenames in train;
    # Optimize hyperparameter values by testing on filenames in validate;
    # Compute error on the filenames in test.

    nd = length(distances)
    nk = 4
    min_logγs       = zeros(nd,nk)
    validate_errors = zeros(nd,nk)
    test_errors     = zeros(nd,nk)

    for k in 1:nk, (i, d) in enumerate(distances)
        min_logγ, min_error_validate, test_error = get_min_gamma(k, d, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=log_γs)
        min_logγs[i,k]       = min_logγ
        validate_errors[i,k] = min_error_validate
        test_errors[i,k]     = test_error
    end

    # # for rational quadratic kernel, have 2 hyperparameters to optimize
    # k=5
    # for (i, d) in enumerate(distances)
    #     min_logγ, min_error_validate, test_error = get_min_gamma_alpha(k, d, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=log_γs)
    #     min_logγs[i,5]       = min_logγ
    #     validate_errors[i,5] = min_error_validate
    #     test_errors[i,5]     = test_error
    # end

    println("MIN LOG γs")
    println(min_logγs)
    println("VALIDATE Mean Error")
    println(validate_errors)
    println("TEST Mean Error")
    println(test_errors)

    a = argmin(test_errors)
    d = a[1]
    k = a[2]
    logγ    = min_logγs[d,k]
    kernel  = get_kernel(k, logγ, 0.0, distances[d])
    ℳ      = model(𝒟_train; kernel=kernel)
    anim    = animate_profile_and_model_output(ℳ, 𝒟_test)
    gif(anim, "$(descriptor)_$(typeof(problem))_$(problem.type)_kernel_$(k)_gamma_$(logγ).gif");

    println("===============")
    println("-- kernel ............. $(k)")
    println("-- norm ............... $(distances[d])")
    println("-- logγ ............... $(logγ)")
    println("-- validate error ..... $(validate_errors[d, k])")
    println("-- test error ......... $(test_errors[d, k])")

    return (min_logγs, validate_errors, test_errors)
end



function get_min_gamma(k, distance, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=-0.3:0.1:0.3)

    errors_validate = zeros(length(log_γs))

    for (i, logγ) in enumerate(log_γs)

        kernel = get_kernel(k, logγ, 0.0, distance)
        ℳ = model(𝒟_train; kernel=kernel)

        # -----compute mean error for true check----
        errors_validate[i] = get_me_true_check(ℳ, 𝒟_validate)
    end

    i                   = argmin(errors_validate)
    min_logγ            = log_γs[i]
    min_error_validate  = errors_validate[i]

    # using the log_γ value that minimizes the error on the validation set,
    # see how the model performs on the test set.
    kernel = get_kernel(k, min_logγ, 0.0, distance)
    ℳ = model(𝒟_train; kernel=kernel);
    error_test = get_me_true_check(ℳ, 𝒟_test)

    return (min_logγ, min_error_validate, error_test)
end

# function get_min_gamma(k::Int64, 𝒟::ProfileData, distance, log_γs)
#     # returns the gamma value that minimizes the mean error on the true check
#     # - only tests the gamma values listed in the log_γs parameter
#
#     mets  = zeros(length(log_γs)) # mean error for each gamma (true check)
#     for (i, logγ) in enumerate(log_γs)
#
#         kernel = get_kernel(k, logγ, 0.0, distance)
#         ℳ = model(𝒟; kernel=kernel);
#
#         # -----compute mean error for true check----
#         mets[i] = get_me_true_check(ℳ, 𝒟)
#     end
#
#     i = argmin(mets)
#     min_logγ = log_γs[i]
#     min_error = mets[i]
#
#     return (min_logγ, min_error)
# end

function get_min_gamma_alpha(k, distance, 𝒟_train, 𝒟_validate, 𝒟_test; log_γs=-0.3:0.1:0.3, log_αs=-0.3:0.1:0.3)
    # returns the gamma value that minimizes the mean error on the true check
    # only tests the gamma values listed in log_γs parameter

    errors_validate = @MArray zeros(length(log_γs), length(log_αs))

    for i in eachindex(log_γs), j in eachindex(log_αs)

        kernel = get_kernel(k, log_γs[i], 0.0, distance; logα=log_αs[j])
        ℳ = model(𝒟_train; kernel=kernel);

        # -----compute mean error for true check----
        errors_validate[i,j] = get_me_true_check(ℳ, 𝒟_validate)
    end

    m = argmin(errors_validate)
    min_logγ = log_γs[m[1]]
    min_logα = log_αs[m[2]]
    min_error_validate = errors_validate[m]

    # using the log_γ value that minimizes the error on the validation set,
    # see how the model performs on the test set.
    kernel = get_kernel(k, min_logγ, 0.0, distance; logα=min_logα)
    ℳ = model(𝒟_train; kernel=kernel);
    error_test = get_me_true_check(ℳ, 𝒟_test)

    return (min_logγ, min_error_validate, error_test)
end



function plot_landscapes_compare_error_metrics(k::Int64, 𝒟::ProfileData, distance, log_γs)
    # Compare mean log marginal likelihood with
    #    mean error on greedy check and
    #    mean error on true check

    mlls = zeros(length(log_γs)) # mean log marginal likelihood
    mes  = zeros(length(log_γs)) # mean error (greedy check)
    mets  = zeros(length(log_γs)) # mean error (true check)

    for i in 1:length(log_γs)

        kernel = get_kernel(k, log_γs[i], 0.0, distance)
        ℳ = model(𝒟; kernel=kernel)

        # -----compute mll loss----
        mlls[i] = -1*mean_log_marginal_loss(𝒟.y_train, ℳ, add_constant=false)

        # -----compute mean error for greedy check (same as in plot log error)----
        mes[i] = get_me_greedy_check(ℳ, 𝒟)

        # -----compute mean error for true check----
        mets[i] = get_me_true_check(ℳ, 𝒟)

    end

    ylims = ( minimum([minimum(mets), minimum(mes)]) , maximum([maximum(mets), maximum(mes)]) )

    mll_plot = plot(log_γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
    vline!([log_γs[argmin(mlls)]])
    mes_plot  = plot(log_γs, mes,  xlabel="log(γ)", title="ME on greedy check, min = $(round(minimum(mes);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([log_γs[argmin(mes)]])
    met_plot  = plot(log_γs, mets,  xlabel="log(γ)", title="ME on true check, min = $(round(minimum(mets);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([log_γs[argmin(mets)]])

    layout = @layout [a; b; c]
    return plot(mll_plot, mes_plot, met_plot, layout = layout)
end


function plot_landscapes_compare_files_me(filenames, k::Int64, distance, log_γs, problem; D=16, N=4)
    # visual comparison of the mean error on true check for every file in filenames

    function get_me(filename)
        𝒟 = data(file, problem; D=D, N=N)

        mes  = zeros(length(log_γs))
        for i in 1:length(log_γs)
            kernel = get_kernel(k, log_γs[i], 0.0, distance)
            ℳ = model(𝒟; kernel=kernel)
            mes[i] = get_me_true_check(ℳ, 𝒟)
        end

        return mes
    end

    results = Dict(file => get_me(file) for file in filenames)

    # put all the data into one array for plotting
    for r in results
        all = hcat(all, r[file])
    end

    layout = (length(filenames), 1)
    ylims = (minimum(all),maximum(all))

    # minimizing γ values
    argmin_logγ = vcat([log_γs[argmin(results[file])]
                for file in filenames])

    titles = ["$(file), log(γ)=$(argmin_logγ[i]), min = $(round(minimum(results[filenames[i]]);digits=5))"
             for i in eachindex(filenames)]

    p = plot(log_γs, xlabel="log(γ)", ylabel="ME, true check", title=titles, legend=false, yscale=:log10, ylims=ylims, layout=layout)  # 1D plot: mean error vs. γ

    vline!(argmin_γ')

    return p
end

function plot_error_histogram(ℳ, 𝒟::ProfileData, time_index)
    # mean error for true check
    gpr_prediction = predict(ℳ, 𝒟; postprocessed=true)
    n = 𝒟.Nt-1

    gpr_error = zeros(n-1)
    for i in 1:n-1
        exact    = 𝒟.vavg[i+1]
        predi    = gpr_prediction[i+1]
        gpr_error[i] = euclidean_distance(exact, predi) # euclidean distance
    end
    mean_error = sum(gpr_error) / n

    error_plot_log = histogram(log.(gpr_error), title = "log(error) at each timestep of the full evolution", xlabel="log(Error)", ylabel="Frequency",ylims=(0,250), label="frequency")
    vline!([log(mean_error)], line = (4, :dash, 0.8), label="mean error")
    vline!([log(gpr_error[time_index])], line = (1, :solid, 0.6), label="error at t=$(time_index)")
end
