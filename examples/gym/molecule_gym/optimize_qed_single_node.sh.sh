#!/bin/bash
clear
python optimize_qed.py \
    --min-atoms 6 \
    --max-atoms 25 \
    --max-num-bonds 100 \
    --num-gpus 1 \
    --num-cpus 34 \
    --local-dir /scratch/dbiagion/ray_results/qed
