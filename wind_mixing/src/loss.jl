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
    fig = Figure()
    ax_top = Axis(fig[1, 1], yscale=log10)
    ax_bot = Axis(fig[2, 1], yscale=log10)

    ax_top.xlabel = "Epochs"
    ax_bot.xlabel = "Epochs"

    ax_top.ylabel = "Losses"
    ax_bot.ylabel = "Losses"

    u_line = CairoMakie.lines!(ax_top, x, losses.u)
    v_line = CairoMakie.lines!(ax_top, x, losses.v)
    T_line = CairoMakie.lines!(ax_top, x, losses.T)

    ∂u∂z_line = CairoMakie.lines!(ax_top, x, losses.∂u∂z)
    ∂v∂z_line = CairoMakie.lines!(ax_top, x, losses.∂v∂z)
    ∂T∂z_line = CairoMakie.lines!(ax_top, x, losses.∂T∂z)

    profile_line = CairoMakie.lines!(ax_bot, x, profile_losses)
    gradient_line = CairoMakie.lines!(ax_bot, x, gradient_losses)

    legend = fig[:,2] = Legend(fig, [u_line, v_line, T_line, ∂v∂z_line, ∂v∂z_line, ∂T∂z_line, profile_line, gradient_line],
                                ["u", "v", "T", "∂u/∂z", "∂v/∂z", "∂T/∂z", "profile", "gradient"])

    trim!(fig.layout)

    save(OUTPUT_PATH, fig)
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