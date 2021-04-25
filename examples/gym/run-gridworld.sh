#!/bin/bash
clear
size=32
rm -rf "gridworld_${size}_policy_checkpoints" "gridworld_${size}_data.db"
python solve_gridworld.py\
    --size $size\
    --num-workers 34\
    --num-mcts-samples 128\
    --log-level debug

