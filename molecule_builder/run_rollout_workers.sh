#! /bin/bash
# Run rollout.py on multiple workers

conda activate path-to-env

num_workers=2

for (( i=1; i<=$num_workers; i++ ))
do
    python rollout.py --id "$i"  &
done