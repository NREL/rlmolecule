#! /bin/bash
# Running multiple workers

conda activate molecule
num_workers=2

for (( c=1; c<=num_workers; c++ ))
do
    python rollout.py --id "$c"  &
done