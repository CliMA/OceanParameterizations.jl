using Interpolations
using QuadGK
using Jacobi
using PyPlot

function polynomial_decomposition(f, xs, ops; w=one, rtol=1e-8)
    x1, x2 = minimum(xs), maximum(xs)
    ys = f.(xs)
    ℑf = interpolate((xs,), ys, Gridded(Linear()))

    numerators = [quadgk(x -> ℑf(x) * op(x) * w(x), x1, x2, rtol=rtol)[1] for op in ops]
    denominators = [quadgk(x -> op(x)^2 * w(x), x1, x2, rtol=rtol)[1] for op in ops]
    cs = numerators ./ denominators

    N = length(ops)
    f′(x) = sum(cs[n] * ops[n](x) for n in 1:N)

    return cs, f′
end

@info "Decomposing into Fourier series..."

xs = range(-π, π, length=101)
f(x) = x/π + 2exp(-3(x+2)^2) -  3exp(-6(x-1)^2)
Ns = (2, 4, 16)

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
plt.savefig("fourier_decomposition.png")
plt.close("all")

@info "Decomposing into Legendre series..."

xs = range(-1, 1, length=101)
f(x) = x + 2exp(-2π*(x+2/π)^2) -  3exp(-6π*(x-1/π)^2)
Ns = (4, 8, 32)

cs, ℑf = Dict(), Dict()
for N in  Ns
    ops = [x -> legendre(x, n) for n in 0:N]
    cs[N], ℑf[N] = polynomial_decomposition(f, xs, ops)
end

plot(xs, f.(xs), label="data")
for N in Ns
    plot(xs, ℑf[N].(xs), label="N=$N")
end
plt.legend()
plt.title("Legendre series decomposition")
plt.savefig("legendre_decomposition.png")
plt.close("all")

@info "Decomposing into Chebyshev series..."

Ns = (4, 8, 16)

cs, ℑf = Dict(), Dict()
for N in  Ns
    w(x) = 1 / √(1 - x^2)
    ops = [x -> chebyshev(x, n) for n in 0:N]
    cs[N], ℑf[N] = polynomial_decomposition(f, xs, ops, w=w, rtol=1e-5)
end

plot(xs, f.(xs), label="data")
for N in Ns
    plot(xs, ℑf[N].(xs), label="N=$N")
end
plt.legend()
plt.title("Chebyshev series decomposition")
plt.savefig("chebyshev_decomposition.png")

