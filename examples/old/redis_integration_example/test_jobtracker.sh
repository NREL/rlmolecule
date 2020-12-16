#!/bin/bash

# This script is meant to run as a background daemon (optionally, with out/err redirection): 
# `bash <name of this script> > /dev/null 2>&1`

# Use consistent redis settings
. ./redis.config

while true
do
  job_ids=$($redis_cmd SMEMBERS "${job_tag}_PENDING_AND_RUNNING")

  for jid in $job_ids
  do
    #echo "Job ID: " $jid
    
    # Example status str: 4202615|rlmol_exp|debug|rlmolecule|36|TIMEOUT|0:0| 
    job_status_str=$(sacct -j $jid --parsable | head -2 | tail -1)
    
    job_status=$(echo $job_status_str | awk -F "|" '{ print $6 }')
    #echo "Status: " $job_status 
  
    if [ "$job_status" = "RUNNING" ] || [ "$job_status" = "PENDING" ] ; then
      #echo "Job ${jid} is still running or pending"  
      :
    else 
  
      # Decide how to classify the job being looked at and move it to appropriate set
      if [ "$job_status" = "COMPLETED" ]; then
        dest_key="${job_tag}_COMPLETED"  
        $redis_cmd SMOVE "${job_tag}_PENDING_AND_RUNNING" ${dest_key} ${jid}
      elif [ "$job_status" = "FAILED" ]; then
        dest_key="${job_tag}_FAILED"  
        $redis_cmd SMOVE "${job_tag}_PENDING_AND_RUNNING" ${dest_key} ${jid}
      elif [ "$job_status" = "TIMEOUT" ]; then
        dest_key="${job_tag}_TIMEOUT"  
        $redis_cmd SMOVE "${job_tag}_PENDING_AND_RUNNING" ${dest_key} ${jid}
      elif [[ "$job_status" == *CANCELLED* ]]; then
        dest_key="${job_tag}_CANCELLED"
        $redis_cmd SMOVE "${job_tag}_PENDING_AND_RUNNING" ${dest_key} ${jid}
      fi
    
    fi 
    
  done 
  sleep 3

done

