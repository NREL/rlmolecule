#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=4:00:00
#SBATCH --job-name az_stability
# --- Policy Trainer ---
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -n 90
#SBATCH -c 4

export WORKING_DIR=/scratch/$USER/rlmolecule
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu
python train_policy.py
EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/rlmolecule/pstjohn/envs/tf2_cpu
python run_mcts.py
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

srun --pack-group=0 \
     --job-name="az-policy" \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &
     
srun --pack-group=1 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"
