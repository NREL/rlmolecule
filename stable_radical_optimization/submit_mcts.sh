#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=00:20:00
#SBATCH --job-name=mcts_q2_debug
#SBATCH --partition=debug
#SBATCH -n 4
#SBATCH -c 18
#SBATCH --output=/scratch/pstjohn/mcts.%j.out

source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2_cpu

srun python run_mcts.py
