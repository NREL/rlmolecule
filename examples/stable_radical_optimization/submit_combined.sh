#!/bin/bash
#SBATCH --account=bpms
#SBATCH --time=4:00:00
#SBATCH --job-name=stable_rad_opt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jlaw@nrel.gov
# --- Policy Trainer ---
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -N 50

export WORKING_DIR=/scratch/${USER}/rlmolecule/stable_radical_optimization/
mkdir -p $WORKING_DIR
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"

model_dir="/projects/rlmolecule/pstjohn/models/"; 
stability_model="$model_dir/20210214_radical_stability_new_data/"
redox_model="$model_dir/20210602_redox_tempo/"
bde_model="$model_dir/20210216_bde_new_nfp/"
config="config/config_eagle_c.yaml"

cat << EOF > "$START_POLICY_SCRIPT"
#!/bin/bash
source $HOME/.bashrc
module use /nopt/nrel/apps/modules/test/modulefiles/
module load cudnn/8.1.1/cuda-11.2
conda activate rlmol
python -u stable_radical_opt.py \
    --train-policy \
    --config="$config" \
    --stability-model="$stability_model" \
    --redox-model="$redox_model" \
    --bde-model="$bde_model" 
EOF

cat << EOF > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source $HOME/.bashrc
conda activate rlmol
python -u stable_radical_opt.py \
    --rollout \
    --config="$config" \
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
     --ntasks-per-node=9 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"

