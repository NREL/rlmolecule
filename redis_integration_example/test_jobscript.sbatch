#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --partition=debug
#SBATCH --output=outerr/%x-%j.out
#SBATCH --error=outerr/%x-%j.err
#SBATCH --qos=high

source activate redis_integration

export JOB_SPECIFIC_PREFIX="${job_tag}-${SLURM_JOB_ID}"
python test_process.py
