using Flux

weights, re = Flux.destructure(Chain(Dense(10, 10, relu), Dense(10, 5)))

# NN = re(fill(1 ./ 1f5, length(weights)))
# NN = re(randn(Float32, length(weights)) ./ 1f5)

NN = re(weights ./ 1f4)
data = [(randn(Float32, 10), randn(Float32, 5)) for i in 1:50]

loss(x, y) = Flux.mse(NN(x), y)

opt = ADAM(1e-4)

@show params(NN)
mean([loss(point[1], point[2]) for point in data])
Flux.train!(loss, Flux.params(NN), data, opt)
mean([loss(point[1], point[2]) for point in data])

weights, _ = Flux.destructure(NN)
W_1, b_1, W_2, b_2 = params(NN)

display(W_1)
display(b_1)
display(W_2)
display(b_2)

NN(rand(10)) .- NN(rand(10))