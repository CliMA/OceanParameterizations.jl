using DiffEqFlux, DifferentialEquations, Optim, Flux, Zygote, Test, BenchmarkTools

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,ROCK4(),saveat=t))

fastdudt2 = FastChain((x,p) -> x.^3,
            FastDense(2,50,tanh),
            FastDense(50, 2))

fast_n_ode = NeuralODE(fastdudt2,tspan,ROCK4(),saveat=t)

function fast_predict_n_ode(p)
  fast_n_ode(u0,p)
end

function fast_loss_n_ode(p)
    pred = fast_predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

dudt2 = Chain(x -> x.^3,
            Dense(2,50,tanh),
            Dense(50, 2))

n_ode = NeuralODE(dudt2,tspan,ROCK4(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

p = fast_n_ode.p
_p,re = Flux.destructure(dudt2)

@test fastdudt2(ones(2),_p) ≈ dudt2(ones(2))
@test fast_loss_n_ode(p)[1] ≈ loss_n_ode(p)[1]
@test Zygote.gradient((p)->fast_loss_n_ode(p)[1], p)[1] ≈ Zygote.gradient((p)->loss_n_ode(p)[1], p)[1]

@btime Zygote.gradient((p)->fast_loss_n_ode(p)[1], p)
@btime Zygote.gradient((p)->fast_loss_n_ode(p)[1], p)
@btime Zygote.gradient((p)->loss_n_ode(p)[1], p)
@btime Zygote.gradient((p)->loss_n_ode(p)[1], p)

@benchmark Zygote.gradient((p)->fast_loss_n_ode(p)[1], p)
@benchmark Zygote.gradient((p)->loss_n_ode(p)[1], p)