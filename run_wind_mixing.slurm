#!/bin/bash
#SBATCH -J CLIMA_TEST_35
#SBATCH -o CLIMA_TEST_35_%j.out
#SBATCH -e CLIMA_TEST_35_%j.err
#SBATCH --mail-user=xinkai@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --exclusive
module load julia
module load spack
module load gcc/8.3.0 # to get libquadmath
module load julia/1.4.1
module load cuda/10.2
module load openmpi/3.1.4-pmi-cuda

cd /home/xinkai/ClimateSurrogates.jl
unbuffer julia --project wind_mixing/wind_mixing.jl
