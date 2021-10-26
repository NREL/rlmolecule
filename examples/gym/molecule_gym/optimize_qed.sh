#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=qed
#SBATCH --time=4:00:00

##HEAD RAY NODE
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

##ROLLOUT NODES
#SBATCH hetjob 
#SBATCH --tasks-per-node=1
#SBATCH --nodes=4

##TRAINING NODE (GPU)
#SBATCH hetjob
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -x

source env.sh
env # for debugging purposes

# Get nodes
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)
rollout_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1)
rollout_nodes_array=( $rollout_nodes )
learner_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_2)
echo "head node    : "$head_node
echo "rollout nodes: "$rollout_nodes
echo "learner node : "$learner_node

rollout_node_num=$(( $SLURM_JOB_NUM_NODES_HET_GROUP_1 ))
rollout_num_cpus=$(( $rollout_node_num * $SLURM_CPUS_ON_NODE ))
echo "rollout num cpus: "$rollout_num_cpus

ip_prefix=$(srun --pack-group=0 --nodes=1 --ntasks=1 -w $head_node hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
echo "ip_prefix: "$ip_pref
echo "suffix: "$suffix
echo "ip_head: "$ip_head
echo "redis_password: "$redis_password

export ip_head # Exporting for latter access by trainer.py.  From rllib example.

echo "starting head node"
srun --pack-group=0 \
     --nodes=1 \
     --ntasks=1 \
     --job-name="ray-head" \
     -w $head_node \
     ray start --block --head --redis-password=$redis_password &
sleep 10

echo "starting rollout nodes"
for ((  i=0; i<$rollout_node_num; i++ ))
do
  rollout_node=${rollout_nodes_array[$i]}
  echo "starting rollout_node=$rollout_node"
  srun --pack-group=1 \
       --nodes=1 \
       --ntasks=1 \
       --job-name="ray-worker-$i" \
       -w $rollout_node \
       ray start --block --address=$ip_head --redis-password=$redis_password &
  sleep 10
done

echo "starting gpu learner node"
srun --pack-group=2 \
     --nodes=1 \
     --gres=gpu:1 \
     --job-name="ray-gpu-learner" \
     -w $learner_node \
     ray start --block --address=$ip_head --redis-password=$redis_password &

sleep 60

echo "calling PPO train script"
python -u optimize_qed.py \
     --max-atoms 25 \
     --min-atoms 6 \
     --max-num-bonds 100 \
     --max-num-actions 64 \
     --redis-password $redis_password \
     --num-cpus $rollout_num_cpus \
     --num-gpus 1 \
     --local-dir "/scratch/$USER/ray_results/qed"

