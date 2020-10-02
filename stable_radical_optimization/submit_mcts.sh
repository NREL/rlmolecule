#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=00:20:00
#SBATCH --job-name=mcts_q2_debug
#SBATCH --partition=debug
#SBATCH -n 72
#SBATCH -c 1
#SBATCH --output=/scratch/pstjohn/rlmolecule/mcts.%j.out

source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2_cpu

python initialize.py
srun python run_mcts.py
