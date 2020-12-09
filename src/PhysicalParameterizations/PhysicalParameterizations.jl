module PhysicalParameterizations

using OceanTurb
using OceanParameterizations.DataWrangling

include("k_profile_parameterization.jl")
include("turbulent_kinetic_energy_closure.jl")

end