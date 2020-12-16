#!/bin/bash

cd ~/rlmolecule/redis_integration_example

# Use consistent redis settings
. ./redis.config

username=dduplyak
allocation=rlmolecule

# Submit sbatch script that will do actual work

jobid=$(sbatch --parsable -A ${allocation} -J ${job_tag} --export=job_tag="${job_tag}" test_jobscript.sbatch)
echo "Submitted job's ID: $jobid"

# Job accounting using Redis

key="${job_tag}_ALL"
$redis_cmd SADD ${key} $jobid
echo "Recorded this job in Redis, under set: ${key}"

key="${job_tag}_PENDING_AND_RUNNING"
$redis_cmd SADD ${key} $jobid
echo "Recorded this job in Redis, under set: ${key}"
