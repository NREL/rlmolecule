# Previous SLURM scripts

Some example SLURM scripts for potentially related projects.

split_gpus.sh: runs a different model on each GPU for a bunch of nodes. Likely how we'd want to configure the training or serving jobs.

split_nodes.sh: slices nodes in half (with -c 18) and just uses srun to run the same python script on each half-node. Probably pretty similar to how we'll want to run the rollout code
