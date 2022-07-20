#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=rlmolecule
#SBATCH --time=0:10:00
#SBATCH --job-name=test_stable_rad_opt
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4

export WORKING_DIR=/scratch/${USER}/rlmolecule/stable_radical_optimization
mkdir -p $WORKING_DIR

export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"
export START_TFSERVING_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.tfserving.sh"

# The following path needs to point to a shared directory, 
# which will be accessible from both tfserving node and the rollout workers 
export TFSERVING_HOSTNAME_PATH="$WORKING_DIR/tfserving_hostname.$SLURM_JOB_ID"

# make sure the base folder of the repo is on the python path
export PYTHONPATH="$(readlink -e ../../):$PYTHONPATH"

model_dir="/projects/rlmolecule/pstjohn/models/"; 
stability_model="$model_dir/20210214_radical_stability_new_data/"
redox_model="$model_dir/20210214_redox_new_data/"
bde_model="$model_dir/20210216_bde_new_nfp/"

cat << EOF > "$START_POLICY_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu
python -u stable_radical_opt.py --train-policy \
    --stability-model="$stability_model" \
    --redox-model="$redox_model" \
    --bde-model="$bde_model" 
EOF

cat << EOF > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh; 
conda activate /projects/rlmolecule/pstjohn/envs/tf2_cpu
tfserving_hostname=$(cat $TFSERVING_HOSTNAME_PATH)
python -u stable_radical_opt.py --rollout \
    --stability-model="$stability_model" \
    --redox-model="$redox_model" \
    --bde-model="$bde_model" \
    --tfserving_hostname=$tfserving_hostname
EOF

tfserving_redox_model="$model_dir/20210216_redox_tfserving/"
batch_config="./config/tfserving_batch.config"
tfserving_img="/projects/rlmolecule/pstjohn/containers/tensorflow-serving-gpu.simg"
cat << EOF > "$START_TFSERVING_SCRIPT"
#!/bin/bash
# This saves hostname to a file, which can be read from rollout workers 
hostname > $TFSERVING_HOSTNAME_PATH
source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu
module load singularity-container
SINGULARITYENV_MODEL_NAME=reward \
      singularity exec --nv \
      -B "$tfserving_redox_model":/models/reward \
      -B "$batch_config":/models/tfserving_batch.config \
      "$tfserving_img" \
      tf_serving_entrypoint.sh \
      --enable_batching \
      --batching_parameters_file=/models/tfserving_batch.config
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT" "$START_TFSERVING_SCRIPT"

# there are 36 cores on eagle nodes.

# run tfserving
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     --output=$WORKING_DIR/tfserving.%j.out \
     "$START_TFSERVING_SCRIPT" &

# run one policy training job
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
    --output=$WORKING_DIR/gpu.%j.out \
    "$START_POLICY_SCRIPT" &

# and run 7 cpu rollout jobs
srun --gres=gpu:0 --ntasks=7 --cpus-per-task=4 \
    --output=$WORKING_DIR/mcts.%j.out \
    "$START_ROLLOUT_SCRIPT"

