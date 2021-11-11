#!/bin/bash
clear
srun python optimize_rad.py \
    --min-atoms 4 \
    --max-atoms 12 \
    --max-num-bonds 30 \
    --num-gpus 1 \
    --num-cpus 34 \
    --local-dir /scratch/$USER/ray_results/rad
