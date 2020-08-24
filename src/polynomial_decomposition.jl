using Interpolations
using QuadGK
using PyPlot

function polynomial_decomposition(f, xs, ops, rtol=1e-8)
    x1, x2 = minimum(xs), maximum(xs)
    ys = f.(xs)
    ℑf = interpolate((xs,), ys, Gridded(Linear()))

    numerators = [quadgk(x -> ℑf(x) * op(x), x1, x2, rtol=rtol)[1] for op in ops]
    denominators = [quadgk(x -> op(x)^2, x1, x2, rtol=rtol)[1] for op in ops]
    cs = numerators ./ denominators

    N = length(ops)
    f′(x) = sum(cs[n] * ops[n](x) for n in 1:N)

    return cs, f′
end

xs = range(-π, π, length=101) |> collect
f(x) = x/π + 2exp(-3(x+2)^2) -  3exp(-6(x-1)^2)

Ns = (2, 8, 32)

cs_even, ℑf_even = Dict(), Dict()
cs_odd,  ℑf_odd  = Dict(), Dict()
for N in  Ns
    ops_cos = [x -> cos(n*x) for n in 1:N]
    ops_sin = [x -> sin(n*x) for n in 1:N]
    cs_even[N], ℑf_even[N] = polynomial_decomposition(f, xs, ops_cos)
    cs_odd[N],  ℑf_odd[N]  = polynomial_decomposition(f, xs, ops_sin)
end

plot(xs, f.(xs), label="data")
for N in Ns
    plot(xs, ℑf_even[N].(xs) .+ ℑf_odd[N].(xs), label="N=$N")
end
plt.legend()
plt.title("Fourier series decomposition")

