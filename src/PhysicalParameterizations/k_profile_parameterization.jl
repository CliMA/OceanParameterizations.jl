# @Base.kwdef struct Parameters{T<:AbstractFloat} <: AbstractParameters
#     CSL   :: T  = 0.1   # Surface layer fraction
#     Cτ    :: T  = 0.4   # Von Karman constant
#     CNL   :: T  = 6.33  # Non-local flux proportionality constant
#
#     Cstab :: T  = 2.0   # Stable buoyancy flux parameter for wind-driven turbulence
#     Cunst :: T  = 6.4   # Unstable buoyancy flux parameter for wind-driven turbulence
#
#        Cn :: T  = 1.0   # Exponent for effect of stable buoyancy forcing on wind mixing
#     Cmτ_U :: T  = 0.25  # Exponent for effect of unstable buoyancy forcing on wind mixing of U
#     Cmτ_T :: T  = 0.5   # Exponent for effect of unstable buoyancy forcing on wind mixing of T
#     Cmb_U :: T  = 1/3   # Exponent for the effect of wind on convective mixing of U
#     Cmb_T :: T  = 1/3   # Exponent for effect of wind on convective mixing of T
#
#     Cd_U  :: T  = 0.5   # Wind mixing regime threshold for momentum
#     Cd_T  :: T  = 2.5   # Wind mixing regime threshold for tracers
#
#     Cb_U  :: T  = 0.599 # Buoyancy flux parameter for convective turbulence
#     Cb_T  :: T  = 1.36  # Buoyancy flux parameter for convective turbulence
#     Cτb_U :: T  = (Cτ / Cb_U)^(1/Cmb_U) * (1 + Cunst*Cd_U)^(Cmτ_U/Cmb_U) - Cd_U  # Wind stress parameter for convective turbulence
#     Cτb_T :: T  = (Cτ / Cb_T)^(1/Cmb_T) * (1 + Cunst*Cd_T)^(Cmτ_T/Cmb_T) - Cd_T  # Wind stress parameter for convective turbulence
#
#     CRi   :: T  = 0.3   # Critical bulk Richardson number
#     CKE   :: T  = 4.32  # Unresolved turbulence parameter
#     CKE₀  :: T  = 1e-11 # Minimum unresolved turbulence kinetic energy
#
#     KU₀   :: T  = 1e-6  # Interior viscosity for velocity
#     KT₀   :: T  = 1e-7  # Interior diffusivity for temperature
#     KS₀   :: T  = 1e-9  # Interior diffusivity for salinity
# end


"""
Adapted from
https://github.com/sandreza/OceanConvectionUQSupplementaryMaterials/blob/master/src/ForwardMap/fm.jl
updated for latest version of OceanTurb
"""

"""
closure_kpp_full_evolution(parameters, N, Δt, les::LESbraryData; subsample = 1, grid = 1)

Constructs forward map. Assumes initial conditions and boundary conditions are taken from les data.

# Arguments
- `N`: number of gridpoints to output to
- `Δt`: time step size in seconds
- `les`: les data of the LESbraryData type

# Keyword Arguments
- `subsample`: indices to subsample in time,
- `grid`: in case one wants to save the model grid

# Output
- The forward map. A function that takes parameters and outputs temperature profiles
-   `𝑪`: parameters in KPP, assumes that
    𝑪[1]: Surface Layer Fraction
    𝑪[2]: Nonlocal Flux Amplitude
    𝑪[3]: Diffusivity Amplitude
    𝑪[4]: Shear Constant
"""
function closure_kpp_full_evolution(parameters, T⁰, les; subsample = 1, grid = 1)

     # set parameters
     # parameters = KPP.Parameters( CSL = 𝑪[1], CNL = 𝑪[2], Cb_T = 𝑪[3], CKE = 𝑪[4])

     # assume constant interval between time steps
     Δt = les.t[2] - les.t[1]

     # number of gridpoints
     N = length(T⁰)

     # Build the model with a Backward Euler timestepper
     constants = Constants(Float64; α = les.α , β = les.β, f=les.f⁰, g=les.g)
     model = KPP.Model(grid = UniformGrid(N, les.L), stepper=:BackwardEuler, constants = constants, parameters = parameters)

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

        # get average of initial condition of LES
        # T⁰ = coarse_grain(les.T⁰, N, Oceananigans.Grids.Face)

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
