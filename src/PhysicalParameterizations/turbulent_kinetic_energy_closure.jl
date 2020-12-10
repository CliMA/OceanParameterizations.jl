function closure_tke_full_evolution(parameters, N, Δt, les; subsample = 1, grid = 1)

     # set parameters
     # parameters = TKEMassFlux.TKEParameters( Cᴰ = Cᴰ )

     # Build the model with a Backward Euler timestepper
     constants = Constants(Float64; α = les.α , β = les.β, f=les.fᶿ, g=les.g)
     model = TKEMassFlux.Model(grid = UniformGrid(N, les.L), stepper=:BackwardEuler, constants = constants, tke_equation = parameters)

     # Get grid if necessary
     if grid != 1
         zp = collect(model.grid.zc)
         @. grid  = zp
     end

     # Set boundary conditions
     model.bcs.u.top    = FluxBoundaryCondition(u_top)
     model.bcs.u.bottom = FluxBoundaryCondition(u_bottom)
     model.bcs.b.top    = FluxBoundaryCondition(θ_top)
     model.bcs.b.bottom = FluxBoundaryCondition(θ_bottom) # may need to fix

    # define the closure
    function evolve()

        # get average of initial condition of LES
        T⁰ = custom_avg(les.T⁰, N)

        # set equal to initial condition of parameterization
        model.solution.T[1:N] = copy(T⁰)

        # set aside memory
        if subsample != 1
            time_index = subsample
        else
            time_index = 1:length(les.t)
        end
        Nt = length(les.t[time_index])
        𝒢 = zeros(N, Nt)

        # loop the model
        ti = collect(time_index)
        for i in 1:Nt
            t = les.t[ti[i]]
            run_until!(model, Δt, t)
            @. 𝒢[:,i] = model.solution.T[1:N]
        end
        return 𝒢
    end
    return evolve
end
