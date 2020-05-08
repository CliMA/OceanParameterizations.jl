include("upright_convection.jl")

T, zC, t, Nz, Lz, constants, Q, FT, ∂T∂z = load_data("free_convection_profiles.jld2")

# plot_LES_figure(T, zC, t)
# animate_LES_solution(T, zC, t)

# OceanTurb parameters
N = 16
L = Lz
Δt = 60
T₀ = T[1, :]

Nt, _ = size(T)
T_cs = T_coarse_grained = zeros(Nt, N)
for n in 1:Nt
    T_cs[n, :] .= coarse_grain(T[n, :], N)
end

model_args = (constants, N, L, Δt, t, T₀, FT, ∂T∂z)
CSL, CNL, Cb_T, CKE, solutions = [], [], [], [], []

samples = 10
for n in 1:samples
    @info "Sample $n/$samples"
    trace = do_inference(free_convection_model, model_args, T_coarse_grained, iters=100)

    push!(CSL, trace[:CSL])
    push!(CNL, trace[:CNL])
    push!(Cb_T, trace[:Cb_T])
    push!(CKE, trace[:CKE])

    global zc
    sol, zc = trace.retval
    push!(solutions, sol)
end

bson_filename = "inferred_KPP_parameters.bson"
@info "Saving $bson_filename..."

data = Dict(:CSL => CSL, :CNL => CNL, :Cb_T => Cb_T, :CKE => CKE,
            :T => T, :zC => zC, :solutions => solutions, :zC_cs => zc)
bson(bson_filename, data)
