#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=rlmolecule
#SBATCH --time=1:00:00
#SBATCH --job-name=test_stable_rad_opt_local
#SBATCH --nodes=1
#SBATCH --ntasks=1

conda activate rlmol

python stable_radical_opt.py --config=config_local.yaml