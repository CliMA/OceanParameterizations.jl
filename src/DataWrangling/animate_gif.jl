function animate_gif(xs, y, t, x_str; x_label=["" for i in 1:length(xs)], filename=x_str, header="", directory="Output")
    filepath = pwd() * "/" * directory * "/"
    isdir(dirname(filepath)) || mkpath(filepath)

    anim = @animate for n in 1:length(t)
        x_max = maximum([maximum(x) for x in xs])
        x_min = minimum([minimum(x) for x in xs])

        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom, size=(400,400))
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title=header*"t = $(round(t[n]/86400, digits=1)) days", linewidth=3)
        end

        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end

    gif(anim, pwd() * "/$(directory)/$(filename).gif", fps=20)
end
