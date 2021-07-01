using WindMixing: plot_loss


FILE_NAME = "NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8"
DATA_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
OUTPUT_PATH = "C:\\Users\\xinle\\Documents\\OceanParameterizations.jl"

file = jldopen(DATA_PATH, "r")
losses = (
    u = file["losses/u"],
    v = file["losses/v"],
    T = file["losses/T"],
    ∂u∂z = file["losses/∂u∂z"],
    ∂v∂z = file["losses/∂v∂z"],
    ∂T∂z = file["losses/∂T∂z"],
)

train_files = file["training_info/train_files"]

diurnal = occursin("diurnal", train_files[1])

train_parameters = file["training_info/parameters"]
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

loss_scalings = (u=1f0, v=1f0, T=1f0, ∂u∂z=1f0, ∂v∂z=1f0, ∂T∂z=1f0)

close(file)

plot_loss(losses, joinpath(OUTPUT_PATH, "$(FILE_NAME)_loss.png"))
