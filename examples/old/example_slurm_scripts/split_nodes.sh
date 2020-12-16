#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=04:00:00
#SBATCH --job-name=redox_short
#SBATCH --qos=high
#SBATCH -n 10
#SBATCH -c 18
#SBATCH --no-kill
#SBATCH --output=/projects/rlmolecule/pstjohn/redox_calculations/job_output/gaussian.%j.out

newgrp g09
source /home/pstjohn/.bashrc
conda activate /projects/cooptimasoot/pstjohn/envs/rdkit_cpu
ulimit -c 0
cd /scratch/pstjohn

srun python /home/pstjohn/Research/20200608_redox_calculations/gaussian_redox_runner.py
