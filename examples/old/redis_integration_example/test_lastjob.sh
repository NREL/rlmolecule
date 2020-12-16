#!/bin/bash

# Use consistent redis settings
. ./redis.config

# Print all keys/values recorded under the last (most recent) job
job_ids=$($redis_cmd SMEMBERS "${job_tag}_ALL")

lastjob=$(echo $job_ids | sed "s/\ /\n/g" | sort -g | tail -1)

lastjob_keys=$($redis_cmd KEYS "${job_tag}-${lastjob}:*")

for key in $lastjob_keys
do
  echo "Key: " $key
  echo "Value:" $($redis_cmd GET ${key})
  echo "---"
done 
