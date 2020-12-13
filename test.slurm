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
cd /nobackup/users/xinkai
date
unbuffer julia -E '1+1'
date
