#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=4:00:00
#SBATCH --job-name=stable_rad_opt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jlaw@nrel.gov
# --- Policy Trainer ---
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -N 10

export WORKING_DIR=/scratch/${USER}/rlmolecule/stable_radical_optimization/
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
export PYTHONPTATH="/home/jlaw/projects/arpa-e/rlmolecule_fork:$PYTHONPTATH"

model_dir="/projects/rlmolecule/pstjohn/models/"; 
stability_model="$model_dir/20210214_radical_stability_new_data/"
redox_model="$model_dir/20210214_redox_new_data/"
bde_model="$model_dir/20210216_bde_new_nfp/"

cat << EOF > "$START_POLICY_SCRIPT"
#!/bin/bash
export PYTHONPTATH="/home/jlaw/projects/arpa-e/rlmolecule_fork:$PYTHONPTATH"
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu
python -u stable_radical_opt.py --train-policy \
    --stability-model="$stability_model" \
    --redox-model="$redox_model" \
    --bde-model="$bde_model" 
EOF

cat << EOF > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
export PYTHONPTATH="/home/jlaw/projects/arpa-e/rlmolecule_fork:$PYTHONPTATH"
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
conda activate /projects/rlmolecule/pstjohn/envs/tf2_cpu
python -u stable_radical_opt.py --rollout \
    --stability-model="$stability_model" \
    --redox-model="$redox_model" \
    --bde-model="$bde_model" 
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

srun --pack-group=0 \
     --job-name="az-policy" \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

srun --pack-group=1 \
     --ntasks-per-node=6 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"

