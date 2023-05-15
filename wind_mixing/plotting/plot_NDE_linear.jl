using CairoMakie
using FileIO
using JLD2
using Statistics
using LsqFit

uws = -3f-4:-1f-4:-5f-4
wbs = 1f-8:1f-8:3f-8

α = 2f-4
g = 9.80665f0

wTs = wbs ./ (α*g)

T_uws = [zeros()]

uw_dirs = ["test_linear_uw-0.0003_vw0_wT0", 
           "test_linear_uw-0.00040000002_vw0_wT0", 
           "test_linear_uw-0.0005_vw0_wT0"]
uw_NN_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/NN_oceananigans.jld2") for dir in uw_dirs]
uw_baseline_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/baseline_oceananigans.jld2") for dir in uw_dirs]


# stop_time = (length(keys(uw_NN_files[1]["timeseries/T"])) - 2) * 10
stop_time = 11530
times = [uw_NN_files[1]["timeseries/t/$time"] for time in 0:10:stop_time]

uw_NN_mlds = [zeros(length(times)) for file in uw_NN_files]
uw_baseline_mlds = [zeros(length(times)) for file in uw_baseline_files]

for (i, mlds) in pairs(uw_NN_mlds)
    file = uw_NN_files[i]
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findlast(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀)
        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind+1])
        else
            mlds[j] = 0
        end
    end
end

for (i, mlds) in pairs(uw_baseline_mlds)
    file = uw_baseline_files[i]
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findlast(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀)
        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind+1])
        else
            mlds[j] = 0
        end
    end
end

##
fig = Figure()
ax = fig[1,1] = Axis(fig)
# scatter!(ax, uw_files[1]["timeseries/T/11530"][1, 1, :], uw_files[1]["grid/zC"][2:end-1])
# lines!(ax, f["timeseries/T/1000"][1, 1, :], f["grid/zC"][2:end-1])
scatter!(ax, times, uw_NN_mlds[1], label=L"$\overline{u\prime w\prime} = 3 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times, uw_NN_mlds[2], label=L"$\overline{u\prime w\prime} = 4 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times, uw_NN_mlds[3], label=L"$\overline{u\prime w\prime} = 5 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, times, uw_baseline_mlds[1], label=L"$\overline{u\prime w\prime} = 3 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times, uw_baseline_mlds[2], label=L"$\overline{u\prime w\prime} = 4 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times, uw_baseline_mlds[3], label=L"$\overline{u\prime w\prime} = 5 \times 10^{-4}$ m$^2$ s$^{-2}$")
axislegend(ax, position=:rb)
display(fig)
##

wT_dirs = ["test_linear_uw0_vw0_wT5.0985814e-6", 
           "test_linear_uw0_vw0_wT1.0197163e-5", 
           "test_linear_uw0_vw0_wT1.5295744e-5"]

wT_NN_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/NN_oceananigans.jld2") for dir in wT_dirs]
wT_baseline_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/baseline_oceananigans.jld2") for dir in wT_dirs]

wT_NN_mlds = [zeros(length(times)) for file in wT_NN_files]
wT_baseline_mlds = [zeros(length(times)) for file in wT_NN_files]

for (i, mlds) in pairs(wT_NN_mlds)
    file = wT_NN_files[i]
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findlast(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀)

        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind+1])
        else
            mlds[j] = 0
        end
    end
end

for (i, mlds) in pairs(wT_baseline_mlds)
    file = wT_baseline_files[i]
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findlast(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀)

        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind+1])
        else
            mlds[j] = 0
        end
    end
end

##
fig = Figure()
ax = fig[1,1] = Axis(fig)
# scatter!(ax, uw_files[1]["timeseries/T/11530"][1, 1, :], uw_files[1]["grid/zC"][2:end-1])
# lines!(ax, f["timeseries/T/1000"][1, 1, :], f["grid/zC"][2:end-1])
scatter!(ax, wT_NN_mlds[1], label=L"$\overline{w\prime T\prime} = 0.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_NN_mlds[2], label=L"$\overline{w\prime T\prime} = 1 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_NN_mlds[3], label=L"$\overline{w\prime T\prime} = 1.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_baseline_mlds[1], label=L"$\overline{w\prime T\prime} = 0.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_baseline_mlds[2], label=L"$\overline{w\prime T\prime} = 1 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_baseline_mlds[3], label=L"$\overline{w\prime T\prime} = 1.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
axislegend(ax, position=:rb)
display(fig)
##

fig = Figure()
ax = fig[1,1] = Axis(fig)
# scatter!(ax, wT_files[1]["timeseries/T/$(stop_time)"][1, 1, :], wT_files[1]["grid/zC"][2:end-1])
scatter!(ax, uw_NN_files[1]["timeseries/T/$(stop_time)"][1, 1, :], wT_NN_files[1]["grid/zC"][2:end-1])
# scatter!(ax, wT_files[2]["timeseries/T/11530"][1, 1, :], wT_files[2]["grid/zC"][2:end-1])
# scatter!(ax, wT_files[3]["timeseries/T/11530"][1, 1, :], wT_files[3]["grid/zC"][2:end-1])
# lines!(ax, f["timeseries/T/1000"][1, 1, :], f["grid/zC"][2:end-1])
display(fig)
##
# sqrt_time(t, p) = p[1] .* t.^p[2]

linear_line(t, p) = p[1] .+ p[2] .* t


p0 = [1e-4, 0.5]

# uw_NN_mlds_fitdata = vcat([log.(mlds[mlds .!= 0]) for (i, mlds) in pairs(uw_NN_mlds)]...)
uw_NN_mlds_fitdata = vcat([log.(mlds[times .<=10^4.5][2:end]) for (i, mlds) in pairs(uw_NN_mlds)]...)

wT_NN_mlds_fitdata = vcat([log.(mlds[mlds .!= 0]) for (i, mlds) in pairs(wT_NN_mlds)]...)

uw_baseline_mlds_fitdata = vcat([log.(mlds[times .<= 10^4.5][2:end]) for (i, mlds) in pairs(uw_baseline_mlds)]...)
wT_baseline_mlds_fitdata = vcat([log.(mlds[mlds .!= 0]) for (i, mlds) in pairs(wT_baseline_mlds)]...)

# uw_NN_tuw_fitdata = vcat([log.(times[mlds .!= 0] .* abs(uws[i])) for (i, mlds) in pairs(uw_NN_mlds)]...)
uw_NN_tuw_fitdata = vcat([log.(times[times .<= 10^4.5][2:end] .* abs(uws[i])) for (i, mlds) in pairs(uw_NN_mlds)]...)

wT_NN_twT_fitdata = vcat([log.(times[mlds .!= 0] .* abs(wTs[i])) for (i, mlds) in pairs(wT_NN_mlds)]...)

uw_baseline_tuw_fitdata = vcat([log.(times[times .<= 10^4.5][2:end] .* abs(uws[i])) for (i, mlds) in pairs(uw_baseline_mlds)]...)
wT_baseline_twT_fitdata = vcat([log.(times[mlds .!= 0] .* abs(wTs[i])) for (i, mlds) in pairs(wT_baseline_mlds)]...)

# uw_fits = [curve_fit(sqrt_time, times, mlds, p0) for mlds in uw_mlds]
uw_NN_fits = [curve_fit(linear_line, log.(times[mlds .!= 0]), log.(mlds[mlds .!= 0]), p0) for mlds in uw_NN_mlds]
wT_NN_fits = [curve_fit(linear_line, log.(times[mlds .!= 0]), log.(mlds[mlds .!= 0]), p0) for mlds in wT_NN_mlds]

uw_baseline_fits = [curve_fit(linear_line, log.(times[mlds .!= 0]), log.(mlds[mlds .!= 0]), p0) for mlds in uw_baseline_mlds]
wT_baseline_fits = [curve_fit(linear_line, log.(times[mlds .!= 0]), log.(mlds[mlds .!= 0]), p0) for mlds in wT_baseline_mlds]

uw_NN_fit = curve_fit(linear_line, uw_NN_tuw_fitdata, uw_NN_mlds_fitdata, p0)
wT_NN_fit = curve_fit(linear_line, wT_NN_twT_fitdata, wT_NN_mlds_fitdata, p0)

uw_baseline_fit = curve_fit(linear_line, uw_baseline_tuw_fitdata, uw_baseline_mlds_fitdata, p0)
wT_baseline_fit = curve_fit(linear_line, wT_baseline_twT_fitdata, wT_baseline_mlds_fitdata, p0)

##
fig = Figure()
# ax = fig[1,1] = Axis(fig, xscale=log10, yscale=log10)
ax = fig[1,1] = Axis(fig)

scatter!(ax, log.(times[2:end]), log.(uw_NN_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(uw_NN_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(uw_NN_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

# scatter!(ax, times[2:end], uw_NN_mlds[1][2:end], label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times[2:end], uw_NN_mlds[2][2:end], label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, times[2:end], uw_NN_mlds[3][2:end], label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

# scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
# scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(uws[1])), uw_NN_fit.param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(uws[2])), uw_NN_fit.param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(uws[3])), uw_NN_fit.param))

# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_NN_fits[1].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_NN_fits[2].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_NN_fits[3].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[1].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[2].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[3].param))

axislegend(ax, position=:rb)
display(fig)
##
fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(uw_baseline_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[1].param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[2].param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[3].param))

axislegend(ax, position=:rb)
display(fig)
##
fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, log.(times[2:end]), log.(wT_baseline_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_baseline_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_baseline_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_baseline_fits[1].param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_baseline_fits[2].param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_baseline_fits[3].param))

axislegend(ax, position=:rb)
display(fig)
##
fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, log.(times[2:end]), log.(wT_NN_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_NN_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_NN_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$")

lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(wTs[1])), wT_NN_fit.param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(wTs[2])), wT_NN_fit.param))
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end] .* abs(wTs[3])), wT_NN_fit.param))

# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_NN_fits[1].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_NN_fits[2].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_NN_fits[3].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[1].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[2].param))
# lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), uw_baseline_fits[3].param))

axislegend(ax, position=:rb)
display(fig)
##
fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, log.(times[2:end]), log.(wT_mlds[1][2:end]), label=L"$\overline{u\prime w\prime} = 3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_mlds[2][2:end]), label=L"$\overline{u\prime w\prime} = 4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, log.(times[2:end]), log.(wT_mlds[3][2:end]), label=L"$\overline{u\prime w\prime} = 5 \times 10^{-4}$ m$^2$ s$^{-2}$")
lines!(ax, log.(times[2:end]), linear_line(log.(times[2:end]), wT_fits[1].param))
axislegend(ax, position=:rb)
display(fig)
##
linewidth = 4

fig = Figure(resolution=(1920, 960), fontsize=30)
ax_baseline = fig[1,1] = Axis(fig, xscale=log10, yscale=log10, title="Baseline parameterization, γ = $(round(uw_baseline_fit.param[2], digits=2))", xlabel="Time (days)", ylabel="Mixed layer depth (m)")
ax_NN = fig[1,2] = Axis(fig, xscale=log10, yscale=log10, title="Neural differential equations, γ = $(round(uw_NN_fit.param[2], digits=2))", xlabel="Time (days)", ylabel="Mixed layer depth (m)")

scatter!(ax_NN, times[2:end]./(24*60^2), uw_NN_mlds[3][2:end], label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)
scatter!(ax_NN, times[2:end]./(24*60^2), uw_NN_mlds[2][2:end], label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)
scatter!(ax_NN, times[2:end]./(24*60^2), uw_NN_mlds[1][2:end], label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)

lines!(ax_NN, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_NN_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[3])).^uw_NN_fit.param[2], linewidth=linewidth)
lines!(ax_NN, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_NN_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[2])).^uw_NN_fit.param[2], linewidth=linewidth)
lines!(ax_NN, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_NN_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[1])).^uw_NN_fit.param[2], linewidth=linewidth)

scatter!(ax_baseline, times[2:end]./(24*60^2), uw_baseline_mlds[3][2:end], label=L"$\overline{u\prime w\prime} = -5 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)
scatter!(ax_baseline, times[2:end]./(24*60^2), uw_baseline_mlds[2][2:end], label=L"$\overline{u\prime w\prime} = -4 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)
scatter!(ax_baseline, times[2:end]./(24*60^2), uw_baseline_mlds[1][2:end], label=L"$\overline{u\prime w\prime} = -3 \times 10^{-4}$ m$^2$ s$^{-2}$", markersize=10)

lines!(ax_baseline, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_baseline_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[3])).^uw_NN_fit.param[2], linewidth=linewidth)
lines!(ax_baseline, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_baseline_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[2])).^uw_NN_fit.param[2], linewidth=linewidth)
lines!(ax_baseline, times[times.<=10^4.5][2:end]./(24*60^2), exp(uw_baseline_fit.param[1]).*(times[times.<=10^4.5][2:end] .* abs(uws[1])).^uw_NN_fit.param[2], linewidth=linewidth)


linkyaxes!(ax_NN, ax_baseline)

hideydecorations!(ax_NN, grid = false)

axislegend(ax_NN, position=:rb)
axislegend(ax_baseline, position=:rb)
display(fig)
save("plots/mld_scaling_uw.pdf", fig, px_per_unit=4, pt_per_unit=4)

#%%
fig = Figure(resolution=(1920, 960), fontsize=30)
ax_baseline = fig[1,1] = Axis(fig, xscale=log10, yscale=log10, title="Baseline parameterization, γ = $(round(wT_baseline_fit.param[2], digits=2))", xlabel="Time (days)", ylabel="Mixed layer depth (m)")
ax_NN = fig[1,2] = Axis(fig, xscale=log10, yscale=log10, title="Neural differential equations, γ = $(round(wT_NN_fit.param[2], digits=2))", xlabel="Time (days)", ylabel="Mixed layer depth (m)")

scatter!(ax_NN, times[2:end]./(24*60^2), wT_NN_mlds[3][2:end], label=L"$\overline{w\prime T\prime} = 1.5 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)
scatter!(ax_NN, times[2:end]./(24*60^2), wT_NN_mlds[2][2:end], label=L"$\overline{w\prime T\prime} = 1 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)
scatter!(ax_NN, times[2:end]./(24*60^2), wT_NN_mlds[1][2:end], label=L"$\overline{w\prime T\prime} = 0.5 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)

lines!(ax_NN, times[2:end]./(24*60^2), exp(wT_NN_fit.param[1]).*(times[2:end] .* abs(wTs[3])).^wT_NN_fit.param[2], linewidth=linewidth)
lines!(ax_NN, times[2:end]./(24*60^2), exp(wT_NN_fit.param[1]).*(times[2:end] .* abs(wTs[2])).^wT_NN_fit.param[2], linewidth=linewidth)
lines!(ax_NN, times[2:end]./(24*60^2), exp(wT_NN_fit.param[1]).*(times[2:end] .* abs(wTs[1])).^wT_NN_fit.param[2], linewidth=linewidth)

scatter!(ax_baseline, times[2:end]./(24*60^2), wT_baseline_mlds[3][2:end], label=L"$\overline{w\prime T\prime} = 1.5 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)
scatter!(ax_baseline, times[2:end]./(24*60^2), wT_baseline_mlds[2][2:end], label=L"$\overline{w\prime T\prime} = 1 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)
scatter!(ax_baseline, times[2:end]./(24*60^2), wT_baseline_mlds[1][2:end], label=L"$\overline{w\prime T\prime} = 0.5 \times 10^{-5}$ $\degree$C m s$^{-1}$", markersize=10)

lines!(ax_baseline, times[2:end]./(24*60^2), exp(wT_baseline_fit.param[1]).*(times[2:end] .* abs(wTs[3])).^wT_NN_fit.param[2], linewidth=linewidth)
lines!(ax_baseline, times[2:end]./(24*60^2), exp(wT_baseline_fit.param[1]).*(times[2:end] .* abs(wTs[2])).^wT_NN_fit.param[2], linewidth=linewidth)
lines!(ax_baseline, times[2:end]./(24*60^2), exp(wT_baseline_fit.param[1]).*(times[2:end] .* abs(wTs[1])).^wT_NN_fit.param[2], linewidth=linewidth)

linkyaxes!(ax_NN, ax_baseline)

hideydecorations!(ax_NN, grid = false)

axislegend(ax_NN, position=:rb)
axislegend(ax_baseline, position=:rb)
display(fig)
save("plots/mld_scaling_wT.pdf", fig, px_per_unit=4, pt_per_unit=4)

#%%
for file in uw_NN_files
    close(file)
end

for file in uw_baseline_files
    close(file)
end

for file in wT_NN_files
    close(file)
end

for file in wT_baseline_files
    close(file)
end
