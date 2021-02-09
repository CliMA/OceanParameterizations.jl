using OceanTurb

using OceanTurb: @add_standard_model_fields, AbstractParameters, AbstractModel

import OceanTurb.KPP: ∂B∂z

const nsol = 4
@solution U V T S

Base.@kwdef struct ModifiedPacanowskiPhilanderParameters{T} <: AbstractParameters
    Cν₀ :: T = 1e-4
    Cν₋ :: T = 1e-1
    Pr  :: T = 1.0
    Riᶜ :: T = 0.25
    ΔRi :: T = 0.1
end

struct ModifiedPacanowskiPhilanderModel{TS, G, T} <: AbstractModel{TS, G, T}
    @add_standard_model_fields
    parameters :: ModifiedPacanowskiPhilanderParameters{T}
    constants  :: Constants{T}
end

function ModifiedPacanowskiPhilanderModel(; N, L,
          grid = UniformGrid(N, L),
     constants = Constants(),
    parameters = Parameters(),
       stepper = :ForwardEuler,
           bcs = BoundaryConditions((ZeroFluxBoundaryConditions() for i=1:nsol)...)
    )

    solution = Solution((CellField(grid) for i=1:nsol)...)
    K = (U=KU, V=KV, T=KT, S=KS)
    R = (U=RU, V=RV, T=RT, S=RS)
    eqn = Equation(K=K, R=R)
    lhs = OceanTurb.build_lhs(solution)

    timestepper = Timestepper(stepper, eqn, solution, lhs)

    return ModifiedPacanowskiPhilanderModel(Clock(), grid, timestepper, solution, bcs, parameters, constants)
end


function local_richardson(U, V, T, S, g, α, β, i)
    Bz = ∂B∂z(T, S, g, α, β, i)
    S² = ∂z(U, i)^2 + ∂z(V, i)^2

    if S² == 0 && Bz == 0 # Alistair Adcroft's theorem
        return 0
    else
        return Bz / S²
    end
end

local_richardson(m, i) = local_richardson(m.solution.U, m.solution.V, m.solution.T,
                                          m.solution.S, m.constants.g, m.constants.α,
                                          m.constants.β, i)

tanh_step(x) = (1 - tanh(x)) / 2

KU(Ri, ν₀, ν₋, Riᶜ, ΔRi) = ν₀ + ν₋ * tanh_step((Ri - Riᶜ) / ΔRi)
KT(Ri, ν₀, ν₋, Riᶜ, ΔRi, Pr) = KU(Ri, ν₀, ν₋, Riᶜ, ΔRi) / Pr

KU(m, i) = KU(local_richardson(m, i), m.parameters.Cν₀, m.parameters.Cν₋,
              m.parameters.Riᶜ, m.parameters.ΔRi)

KT(m, i) = KT(local_richardson(m, i), m.parameters.Cν₀, m.parameters.Cν₋,
              m.parameters.Riᶜ, m.parameters.ΔRi, m.parameters.Pr)

const KV = KU
const KS = KT

RU(m, i) =   m.constants.f * m.solution.V[i]
RV(m, i) = - m.constants.f * m.solution.U[i]
RT(m, i) = 0
RS(m, i) = 0
