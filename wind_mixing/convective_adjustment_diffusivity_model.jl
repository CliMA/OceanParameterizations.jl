using OceanTurb

using OceanTurb: @add_standard_model_fields, AbstractParameters, AbstractModel

import OceanTurb.KPP: ∂B∂z

const nsol = 4
@solution U V T S

Base.@kwdef struct ConvectiveAdjustmentParameters{T} <: AbstractParameters
    κ   :: T = 1
end

struct ConvectiveAdjustmentModel{TS, G, T} <: AbstractModel{TS, G, T}
    @add_standard_model_fields
    parameters :: ConvectiveAdjustmentParameters{T}
    constants  :: Constants{T}
end

function ConvectiveAdjustmentModel(; N, L,
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

    return ConvectiveAdjustmentModel(Clock(), grid, timestepper, solution, bcs, parameters, constants)
end

function temperature_gradient(T, i)
    return ∂z(T, i)
end

temperature_gradient(m, i) = temperature_gradient(m.solution.T, i)

KU() = 0
KT(∂T∂z, κ) = κ * (∂T∂z < 0)

KU(m, i) = KU()
KT(m, i) = KT(temperature_gradient(m, i), m.parameters.κ)

const KV = KU
const KS = KT

RU(m, i) =   m.constants.f * m.solution.V[i]
RV(m, i) = - m.constants.f * m.solution.U[i]
RT(m, i) = 0
RS(m, i) = 0