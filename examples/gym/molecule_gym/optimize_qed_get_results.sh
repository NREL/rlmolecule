#!/bin/bash
# Usage:  optimize_qed_get_results.sh <slurm-file>
# Writes a csv results file, <slurm-file>.top_hits.csv
./optimize_qed_scrape_log.sh $1
python optimize_qed_parse_tophits.py $1.top_hits
