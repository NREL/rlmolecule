#!/bin/bash
clear
size=16
rm -rf "policy_checkpoints" "gridworld_${size}_data.db"
python solve_gridworld.py\
    --size $size\
    --num-workers 34\
    --num-mcts-samples 64\
    --log-level debug

