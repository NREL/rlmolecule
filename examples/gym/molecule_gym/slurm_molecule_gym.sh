#!/bin/bash

#SBATCH --job-name=molecule_gym
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=480
#SBATCH --account=rlmolecule
#SBATCH --cpu-freq=high-high:Performance

# MODIFY HERE according to your environment setup
source ~/arlmol
unset LD_PRELOAD

echo "executing command... python -u -m $@"

python -u -m "$@"

echo "done!"