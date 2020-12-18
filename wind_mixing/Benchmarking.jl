using OceanParameterizations
using WindMixing
using OceanTurb
using BenchmarkTools
using OrdinaryDiffEq

files =  ["free_convection", "strong_wind", "strong_wind_no_coriolis",
            "weak_wind_strong_cooling", "strong_wind_weak_cooling", "strong_wind_weak_heating"]

test_file = "free_convection"

𝒟test = WindMixing.data(test_file)
𝒟train = 𝒟test # just for Benchmarking purposes
les = read_les_output(test_file)

# KPP Parameterization (no training)
#
# parameters = KPP.Parameters() # default parameters
# KPP_model = closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], les)
#
# # Time building the model
# @btime closure_kpp_full_evolution(parameters, 𝒟test.T.coarse[:,1], les) # 86.448 μs (168 allocations: 19.03 KiB)
#
# # Time running the model
# @btime KPP_model() # 25.762 ms (4774 allocations: 242.36 KiB)
#
# ## TKE Parameterization (no training; use default parameters)
#
# # les = read_les_output(test_file)
# parameters = TKEMassFlux.TKEParameters() # default parameters
# TKE_model = closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], les)
#
# # Time building the model
# @btime closure_tke_full_evolution(parameters, 𝒟test.T.coarse[:,1], les) # 92.100 μs (213 allocations: 23.08 KiB)
#
# # Time running the model
# @btime TKE_model() # 2.710 ms (5894 allocations: 259.86 KiB)

## GPR
function build()
    # Set the kernel manually
    uw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
    vw_kernel = get_kernel(1,0.1,0.0,euclidean_distance)
    wT_kernel = get_kernel(1,0.1,0.0,euclidean_distance)

    # Trained GP models
    uw_GP_model = gp_model(𝒟train.uw, uw_kernel)
    vw_GP_model = gp_model(𝒟train.vw, vw_kernel)
    wT_GP_model = gp_model(𝒟train.wT, wT_kernel)

    return (uw_GP_model, vw_GP_model, wT_GP_model)
end

println("Time to build 3 GP models:")
@btime build() # 42.818 ms (148 allocations: 10.07 MiB)

uw_GP_model, vw_GP_model, wT_GP_model = build()
uw_GP = predict(𝒟test.uw, uw_GP_model)
vw_GP = predict(𝒟test.vw, vw_GP_model)
wT_GP = predict(𝒟test.wT, wT_GP_model)

@time predict(𝒟test.wT, wT_GP_model)
println("Time to predict wT:")
println(@time predict(𝒟test.wT, wT_GP_model))


function run_model(uw_GP_model, vw_GP_model, wT_GP_model)
    # GP predictions on test data
    # uw_GP = predict(𝒟test.uw, uw_GP_model)
    # vw_GP = predict(𝒟test.vw, vw_GP_model)
    # wT_GP = predict(𝒟test.wT, wT_GP_model)
    t = 𝒟test.t
    uvT₀      = 𝒟test.uvT_scaled[:,1]
    zF_coarse = 𝒟test.uw.z
    zC_coarse = 𝒟test.u.z
    f⁰        = les.f⁰
    Nz        = 32

    ∂z(vec) = (vec[1:Nz] .- vec[2:Nz+1]) ./ diff(zF_coarse)
    function f(dx, x, p, t)
        u = x[1:Nz]
        v = x[Nz+1:2*Nz]
        dx[1:Nz] .= -∂z(uw_GP_model(x)) .+ f⁰ .* v
        dx[Nz+1:2*Nz] .= -∂z(vw_GP_model(x)) .- f⁰ .* u
        dx[2*Nz+1:end] .= -∂z(wT_GP_model(x))
    end

    prob = ODEProblem(f, uvT₀, (t[1],t[289]), saveat=t)
    sol = solve(prob, ROCK4())

    # split_array(uvT) = (uvT[1:Nz,:], uvT[Nz+1:2*Nz,:], uvT[2*Nz+1:end,:])
    # u_pred, v_pred, T_pred = split_array(sol)
    return sol
end

println("Time to run 3 GP models + solve:")
println(@time run_model(uw_GP_model, vw_GP_model, wT_GP_model))
