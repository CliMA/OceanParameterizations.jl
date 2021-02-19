module PhysicalParameterizations

using OceanTurb
using OceanParameterizations.DataWrangling
using Oceananigans.Grids: Center, Face

export closure_kpp_full_evolution,
       closure_tke_full_evolution

include("k_profile_parameterization.jl")
include("turbulent_kinetic_energy_closure.jl")

end
