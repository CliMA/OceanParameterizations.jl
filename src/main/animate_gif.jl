function animate_gif(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str, dir="Output")
    try
        mkdir(pwd*"/"*dir)
    catch
    end
    anim = @animate for n in 1:size(xs[1],2)
    x_max = maximum(maximum(x) for x in xs)
    x_min = minimum(minimum(x) for x in xs)
        # @info "$x_str frame of $n/$(size(uw,2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n]/86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    gif(anim, pwd()*"/$(dir)/$(filename).gif", fps=20)
end
