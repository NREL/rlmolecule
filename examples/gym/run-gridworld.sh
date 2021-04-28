#!/bin/bash
clear
size=16
rm -rf "gridworld_${size}_policy_checkpoints" "gridworld_${size}_data.db"
python solve_gridworld.py\
    --size $size\
    --num-workers 3\
    --num-mcts-samples 32\
    --log-level debug

