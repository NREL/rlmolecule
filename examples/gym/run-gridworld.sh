#!/bin/bash
rm -rf gridworld_policy_checkpoints gridworld_data.db
clear
python solve_gridworld.py --log-level debug
