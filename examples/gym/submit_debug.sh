#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=rlldrd
#SBATCH --qos=high
#SBATCH --time=1:00:00  # start with 10min for debug
#SBATCH --job-name=gridworld_example_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4

export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
export PYTHONPATH=$PWD/../..:$PYTHONPATH

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
module purge
conda activate /projects/rlmolecule/eskordil/envs/tf2_gpu_cloned
python -u solve_gridworld.py --train-policy --size=32
EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
module purge
conda activate /projects/rlmolecule/eskordil/envs/tf2_cpu_cloned
python -u solve_gridworld.py --rollout --size=32 --num-mcts-samples=256
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

# there are 36 cores on eagle nodes.
# run one policy training job
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     "$START_POLICY_SCRIPT" &

# and run 8 cpu rollout jobs
srun --gres=gpu:0 --ntasks=8 --cpus-per-task=4 \
     "$START_ROLLOUT_SCRIPT"
