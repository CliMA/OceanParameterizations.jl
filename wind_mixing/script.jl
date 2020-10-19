using NCDatasets
using Plots

ds = NCDataset("wind_mixing\\Data\\free_convection_horizontal_averages_25W.nc")

zC = Array(ds["zC"])
zF = Array(ds["zF"])

uT = Array(ds["uT"])
vT = Array(ds["vT"])
wT = Array(ds["wT"])

T = Array(ds["T"])

t = Array(ds["time"])

plot(T[:,end],zC)
