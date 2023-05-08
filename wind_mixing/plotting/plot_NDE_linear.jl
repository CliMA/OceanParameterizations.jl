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
uw_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/NN_oceananigans.jld2") for dir in uw_dirs]

stop_time = (length(keys(uw_files[1]["timeseries/T"])) - 2) * 10
times = [uw_files[1]["timeseries/t/$time"] for time in 0:10:stop_time]

uw_mlds = [zeros(length(keys(file["timeseries/T"]))-1) for file in uw_files]

for (i, mlds) in pairs(uw_mlds)
    file = uw_files[i]
    stop_time = (length(keys(file["timeseries/T"])) - 2) * 10
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀*2)
        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind])
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
scatter!(ax, times, uw_mlds[1], label=L"$\overline{u\prime w\prime} = 3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, times, uw_mlds[2], label=L"$\overline{u\prime w\prime} = 4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, times, uw_mlds[3], label=L"$\overline{u\prime w\prime} = 5 \times 10^{-4}$ m$^2$ s$^{-2}$")
axislegend(ax, position=:rb)
display(fig)
##

wT_dirs = ["test_linear_uw0_vw0_wT5.0985814e-6", 
           "test_linear_uw0_vw0_wT1.0197163e-5", 
           "test_linear_uw0_vw0_wT1.5295744e-5"]

wT_files = [jldopen("../Output/new_nonlocal_NDE_BFGS/$(dir)/NN_oceananigans.jld2") for dir in wT_dirs]
wT_mlds = [zeros(length(keys(file["timeseries/T"]))-1) for file in wT_files]

for (i, mlds) in pairs(wT_mlds)
    file = wT_files[i]
    stop_time = (length(keys(file["timeseries/T"])) - 2) * 10
    ΔT₀ = abs(mean(diff(file["timeseries/T/0"][1, 1, :])))

    for (j, time) in pairs(0:10:stop_time)
        # ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> abs(diff(file["timeseries/T/$time"][1, 1, :])[1])*1.2)
        ind = findfirst(abs.(diff(file["timeseries/T/$time"][1, 1, :])) .> ΔT₀*1.2)

        if !isnothing(ind)
            mlds[j] = abs(file["grid/zC"][2:end-1][ind])
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
scatter!(ax, wT_mlds[1], label=L"$\overline{w\prime T\prime} = 0.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_mlds[2], label=L"$\overline{w\prime T\prime} = 1 \times 10^{-5}$ $\degree$C m s$^{-1}$")
scatter!(ax, wT_mlds[3], label=L"$\overline{w\prime T\prime} = 1.5 \times 10^{-5}$ $\degree$C m s$^{-1}$")
axislegend(ax, position=:rb)
display(fig)
##

fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, wT_files[1]["timeseries/T/11530"][1, 1, :], wT_files[1]["grid/zC"][2:end-1])
# scatter!(ax, wT_files[2]["timeseries/T/11530"][1, 1, :], wT_files[2]["grid/zC"][2:end-1])
# scatter!(ax, wT_files[3]["timeseries/T/11530"][1, 1, :], wT_files[3]["grid/zC"][2:end-1])
# lines!(ax, f["timeseries/T/1000"][1, 1, :], f["grid/zC"][2:end-1])
display(fig)
##
sqrt_time(t, p) = p[1] .* t.^p[2]

p0 = [1e-4, 0.5]

uw_fits = [curve_fit(sqrt_time, times, mlds, p0) for mlds in uw_mlds]
##
fig = Figure()
ax = fig[1,1] = Axis(fig)
scatter!(ax, times, uw_mlds[1], label=L"$\overline{u\prime w\prime} = 3 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, times, uw_mlds[2], label=L"$\overline{u\prime w\prime} = 4 \times 10^{-4}$ m$^2$ s$^{-2}$")
scatter!(ax, times, uw_mlds[3], label=L"$\overline{u\prime w\prime} = 5 \times 10^{-4}$ m$^2$ s$^{-2}$")
lines!(ax, times, sqrt_time(times, uw_fits[1].param))
axislegend(ax, position=:rb)
display(fig)
##
for file in uw_files
    close(file)
end

for file in wT_files
    close(file)
end
