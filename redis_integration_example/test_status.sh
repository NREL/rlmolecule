#!/bin/bash

# Use consistent redis settings
. ./redis.config

# Print all info that might useful, i.e. list members of all job sets
for setname in _ALL _PENDING_AND_RUNNING _COMPLETED _FAILED _CANCELLED _TIMEOUT
do
  echo "Jobs under ${job_tag}${setname}:"
  $redis_cmd SMEMBERS "${job_tag}${setname}"
  echo "---"
done
