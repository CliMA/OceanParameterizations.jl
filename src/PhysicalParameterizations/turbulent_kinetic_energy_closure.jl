function closure_tke_full_evolution(parameters, U⁰, V⁰, T⁰, t, les; subsample = 1, grid = 1)

     # set parameters
     # parameters = TKEMassFlux.TKEParameters( Cᴰ = Cᴰ )

     # assume constant interval between time steps
     Δt = t[2] - t[1]

     # number of gridpoints
     N = length(T⁰)

     # Build the model with a Backward Euler timestepper
     constants = Constants(Float64; α = les.α , β = les.β, f=les.f⁰, g=les.g)
     model = TKEMassFlux.Model(grid = UniformGrid(N, les.L), stepper=:BackwardEuler, constants = constants, tke_equation = parameters)

     # Get grid if necessary
     if grid != 1
         zp = collect(model.grid.zc)
         @. grid  = zp
     end

     # Set boundary conditions
     model.bcs.U.top    = FluxBoundaryCondition(les.u_top)
     model.bcs.U.bottom = GradientBoundaryCondition(les.u_bottom)
     model.bcs.T.top    = FluxBoundaryCondition(les.θ_top)
     model.bcs.T.bottom = GradientBoundaryCondition(les.θ_bottom) # may need to fix

    # define the closure
    function evolve()

        # set equal to initial condition of parameterization
        model.solution.U[1:N] = copy(U⁰)
        model.solution.V[1:N] = copy(V⁰)
        model.solution.T[1:N] = copy(T⁰)

        # set aside memory
        if subsample != 1
            time_index = subsample
        else
            time_index = 1:length(t)
        end
        Nt = length(t[time_index])

        U_evolution = zeros(N, Nt)
        V_evolution = zeros(N, Nt)
        T_evolution = zeros(N, Nt)

        # loop the model
        ti = collect(time_index)
        for i in 1:Nt
            time = t[ti[i]]
            run_until!(model, Δt, time)
            @. U_evolution[:,i] = model.solution.U[1:N]
            @. V_evolution[:,i] = model.solution.V[1:N]
            @. T_evolution[:,i] = model.solution.T[1:N]
        end
        return (U_evolution, V_evolution, T_evolution)
    end
    return evolve
end
