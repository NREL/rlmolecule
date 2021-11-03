#!/bin/bash
# Usage:  ./optimize_qed_scrape_log.sh <slurm-file>
# Writes a new file "<slurm-file>.top_hits"
grep "GraphGymEnv: 0.9[2-9]" $1 > $1.top_hits

