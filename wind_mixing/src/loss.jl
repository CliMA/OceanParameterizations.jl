function loss(a, b)
    return Flux.mse(a, b)
end

@views split_u(uvT, Nz) = uvT[1:Nz, :]
@views split_v(uvT, Nz) = uvT[Nz+1:2Nz, :]
@views split_T(uvT, Nz) = uvT[2Nz+1:3Nz, :]

@views ∂_∂z(profile, D_face) = hcat([D_face * profile[:,i] for i in 1:size(profile, 2)]...)

function calculate_loss_scalings(losses, fractions, train_gradient)
    velocity_scaling = (1 - fractions.T) / fractions.T * losses.T / (losses.u + losses.v)
    profile_loss = velocity_scaling * (losses.u + losses.v) + losses.T
    
    if train_gradient
        velocity_gradient_scaling = (1 - fractions.∂T∂z) / fractions.∂T∂z * losses.∂T∂z / (losses.∂u∂z + losses.∂v∂z)
        gradient_loss = velocity_gradient_scaling * (losses.∂u∂z + losses.∂v∂z) + losses.∂T∂z
        total_gradient_scaling = (1 - fractions.profile) / fractions.profile * profile_loss / gradient_loss
    else
        velocity_gradient_scaling = 0
        gradient_loss = 0
        total_gradient_scaling = 0
    end

    return (   u = velocity_scaling, 
               v = velocity_scaling, 
               T = 1, 
            ∂u∂z = total_gradient_scaling * velocity_gradient_scaling,
            ∂v∂z = total_gradient_scaling * velocity_gradient_scaling,
            ∂T∂z = total_gradient_scaling ) 
end

function apply_loss_scalings(losses, scalings)
    return (
        u = scalings.u .* losses.u,
        v = scalings.v .* losses.v,
        T = scalings.T .* losses.T,
        ∂u∂z = scalings.∂u∂z .* losses.∂u∂z,
        ∂v∂z = scalings.∂v∂z .* losses.∂v∂z,
        ∂T∂z = scalings.∂T∂z .* losses.∂T∂z,
    )
end

@views function loss_per_tstep(a, b)
    return [loss(a[:,i], b[:,i]) for i in 1:size(a, 2)]
end

function plot_loss(losses::NamedTuple, OUTPUT_PATH)
    profile_losses = losses.u .+ losses.v .+ losses.T
    gradient_losses = losses.∂u∂z .+ losses.∂v∂z .+ losses.∂T∂z

    x = 1:length(losses.u)
    fig = Figure(resolution=(1500, 750))
    ax_top = Axis(fig[1, 1], yscale=log10)
    ax_bot = Axis(fig[1, 2], yscale=log10)

    ax_top.xlabel = "Epochs"
    ax_bot.xlabel = "Epochs"

    ax_top.ylabel = "Losses"
    ax_bot.ylabel = "Losses"

    colors = distinguishable_colors(8, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    u_line = CairoMakie.lines!(ax_top, x, losses.u, color=colors[1])
    v_line = CairoMakie.lines!(ax_top, x, losses.v, color=colors[2])
    T_line = CairoMakie.lines!(ax_top, x, losses.T, color=colors[3])

    ∂u∂z_line = CairoMakie.lines!(ax_top, x, losses.∂u∂z, color=colors[4])
    ∂v∂z_line = CairoMakie.lines!(ax_top, x, losses.∂v∂z, color=colors[5])
    ∂T∂z_line = CairoMakie.lines!(ax_top, x, losses.∂T∂z, color=colors[6])

    profile_line = CairoMakie.lines!(ax_bot, x, profile_losses, color=colors[7])
    gradient_line = CairoMakie.lines!(ax_bot, x, gradient_losses, color=colors[8])

    legend_individual = fig[2,1] = Legend(fig, [u_line, v_line, T_line, ∂u∂z_line, ∂v∂z_line, ∂T∂z_line],
                                ["u", "v", "T", "∂u/∂z", "∂v/∂z", "∂T/∂z"], orientation=:horizontal)

    legend_profile = fig[2,2] = Legend(fig, [profile_line, gradient_line],
                                ["profile", "gradient"], orientation=:horizontal)

    colsize!(fig.layout, 1, CairoMakie.Relative(0.5))
    colsize!(fig.layout, 2, CairoMakie.Relative(0.5))

    rowsize!(fig.layout, 2, CairoMakie.Relative(0.05))

    trim!(fig.layout)

    save(OUTPUT_PATH, fig, pt_per_unit=4, px_per_unit=4)
end

function plot_loss(losses, OUTPUT_PATH)
    x = 1:length(losses)
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10)

    ax.xlabel = "Epochs"
    ax.ylabel = "Total Loss"

    line = CairoMakie.lines!(ax, x, losses)

    trim!(fig.layout)

    save(OUTPUT_PATH, fig)
end