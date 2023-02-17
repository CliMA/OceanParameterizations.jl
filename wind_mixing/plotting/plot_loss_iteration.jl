using CairoMakie

FILE_NAME = "NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8"
DATA_PATH = joinpath("..\\extracted_training_output", "$(FILE_NAME)_extracted.jld2")

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

profile_losses = losses.u .+ losses.v .+ losses.T
gradient_losses = losses.∂u∂z .+ losses.∂v∂z .+ losses.∂T∂z

x = 1:length(losses.u)

##
fig = Figure(resolution=(1500, 850), fontsize=30, figure_padding=30)
ax_top = Axis(fig[1, 1], yscale=log10)
ax_bot = Axis(fig[1, 2], yscale=log10)

ax_top.xlabel = "Training iteration"
ax_bot.xlabel = "Training iteration"

ax_top.ylabel = "Losses"
ax_bot.ylabel = "Losses"

colors = distinguishable_colors(8, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

u_line = lines!(ax_top, x, losses.u, color=colors[1], linewidth=5)
v_line = lines!(ax_top, x, losses.v, color=colors[2], linewidth=5)
T_line = lines!(ax_top, x, losses.T, color=colors[3], linewidth=5)

∂u∂z_line = lines!(ax_top, x, losses.∂u∂z, color=colors[4], linewidth=5)
∂v∂z_line = lines!(ax_top, x, losses.∂v∂z, color=colors[5], linewidth=5)
∂T∂z_line = lines!(ax_top, x, losses.∂T∂z, color=colors[6], linewidth=5)

profile_line = lines!(ax_bot, x, profile_losses, color=colors[7], linewidth=5)
gradient_line = lines!(ax_bot, x, gradient_losses, color=colors[8], linewidth=5)

legend_individual = fig[2,1] = Legend(fig, [u_line, v_line, T_line, ∂u∂z_line, ∂v∂z_line, ∂T∂z_line],
                            [L"u", L"v", L"T", L"\frac{\partial u}{\partial z}", L"\frac{\partial v}{\partial z}", L"\frac{\partial T}{\partial z}"], orientation=:horizontal)
legend_individual.tellheight = false

legend_profile = fig[2,2] = Legend(fig, [profile_line, gradient_line],
                            ["Profile", "Gradient"], orientation=:horizontal)

label_a = fig[1, 1, TopLeft()] = Label(fig, "A", fontsize = 30, font = :bold, halign = :right, padding = (0, 25, 0, 0))
label_b = fig[1, 2, TopLeft()] = Label(fig, "B", fontsize = 30, font = :bold, halign = :right, padding = (0, 25, 0, 0))
                            
display(fig)
##
save("plots/loss_iteration.pdf", fig, pt_per_unit=4, px_per_unit=4)