function closure_free_convection_tke_full_evolution(parameters, N, Δt, les;
                                 subsample = 1, grid = 1)

     # set parameters
     # parameters = TKEMassFlux.TKEParameters( Cᴰ = Cᴰ )
     # Build the model with a Backward Euler timestepper
     constants = Constants(Float64; α = les.α , β = les.β, ρ₀= les.ρ, cP=les.cᵖ, f=les.f⁰, g=les.g)
     model = TKEMassFlux.Model(grid = UniformGrid(N, les.L), stepper=:BackwardEuler, constants = constants, tke_equation = parameters)
     # Get grid if necessary
     if grid != 1
         zp = collect(model.grid.zc)
         @. grid  = zp
     end

    # define the closure
    function free_convection()
        # get average of initial condition of LES
        T⁰ = custom_avg(les.T⁰, N)
        # set equal to initial condition of parameterization
        model.solution.T[1:N] = copy(T⁰)
        # Set boundary conditions
        model.bcs.T.top = FluxBoundaryCondition(les.top_T)
        model.bcs.T.bottom = GradientBoundaryCondition(les.bottom_T)
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
    return free_convection
end

function closure_free_convection_tke(parameters, N, Δt, les;
                                 subsample = 1, grid = 1)

     # set parameters
     # parameters = TKEMassFlux.TKEParameters( Cᴰ = Cᴰ )
     # Build the model with a Backward Euler timestepper
     constants = Constants(Float64; α = les.α , β = les.β, ρ₀= les.ρ, cP=les.cᵖ, f=les.f⁰, g=les.g)
     model = TKEMassFlux.Model(grid = UniformGrid(N, les.L), stepper=:BackwardEuler, constants = constants, tke_equation = parameters)
     # Get grid if necessary
     if grid != 1
         zp = collect(model.grid.zc)
         @. grid  = zp
     end

     # Set boundary conditions
     model.bcs.T.top = FluxBoundaryCondition(les.top_T)
     model.bcs.T.bottom = GradientBoundaryCondition(les.bottom_T)
     # set aside memory
     if subsample != 1
         time_index = subsample
     else
         time_index = 1:length(les.t)
     end
     Nt = length(les.t[time_index])
     𝒢 = zeros(N, n_steps+1)

     ti = collect(time_index)

    # define the closure
    function evolve_forward(; T⁰=T⁰, n_steps = 1)
        # get average of initial condition of LES
        T⁰ = custom_avg(T⁰, N)
        # set equal to initial condition of parameterization
        model.solution.T[1:N] = copy(T⁰)

        # loop the model
        for i in 1:n_steps+1
            t = les.t[ti[i]]
            run_until!(model, Δt, t)
            @. 𝒢[:,i] = model.solution.T[1:N]
        end
        return 𝒢
    end
    return evolve_forward
end
