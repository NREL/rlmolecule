#!/bin/bash
clear
size=64 
rm -rf "gridworld_${size}_policy_checkpoints" "gridworld_${size}_data.db"
time python solve_gridworld.py\
        --size $size\
        --num-workers 34\
        --num-mcts-samples 32\
        --log-level debug

