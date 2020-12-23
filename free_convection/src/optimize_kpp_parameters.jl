using Distributions
using LinearAlgebra
using Plots

using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.ParameterDistributionStorage

using Flux: mse

function optimize_kpp_parameters(datasets, true_solutions, T_scaling; ensemble_members, iterations)
    loss(p) = mean(
        mse(T_scaling.(free_convection_kpp(ds, parameters=p)),
            true_solutions[id])
        for (id, ds) in datasets
    )

    n_params = 4
    prior_distributions = [Uniform(0, 1), Uniform(0, 8), Uniform(0, 6), Uniform(0, 5)] .|> Parameterized
    constraints = [[no_constraint()] for _ in 1:n_params]
    prior_names = ["CSL", "CNL", "Cb_T", "CKE"]
    prior = ParameterDistribution(prior_distributions, constraints, prior_names)
    prior_mean = reshape(get_mean(prior), :)
    prior_cov = get_cov(prior) # Assuming parameters are independent.

    initial_ensemble = EKP.construct_initial_ensemble(prior, ensemble_members)
    observed_loss = zeros(1)
    Γ_loss = 1e-8 * ones(1, 1)
    eki = EKP.EKObj(initial_ensemble, observed_loss, Γ_loss, Inversion())
    loss_history = zeros(iterations, ensemble_members)

    for i in 1:iterations
        @info "EKI iteration $i/$iterations..."
        params = eki.u[end]
        loss_i = zeros(ensemble_members, 1)
        for e in 1:ensemble_members
            @info "Computing loss for ensemble member $e/$ensemble_members..."
            p = eki.u[end][e, :]
            p = [clamp(p[1], 0, 1), max(0, p[2]), max(0, p[3]), max(0, p[4])]
            kpp_params = OceanTurb.KPP.Parameters(CSL=p[1], CNL=p[2], Cb_T=p[3], CKE=p[4])
            loss_i[e] = loss_history[i, e] = loss(kpp_params)
            @info "KPP parameters = $p" * @sprintf(", loss = %.12e", loss_i[e])
        end
        EKP.update_ensemble!(eki, loss_i)
    end

    return eki, loss_history
end
