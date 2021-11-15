#!/bin/bash
# Usage:  get_results.sh <slurm-file>
# Writes a csv results file, <slurm-file>.top_hits.csv
./scrape_log.sh $1
python parse_tophits.py $1.top_hits
