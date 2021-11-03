#!/bin/bash
clear
python optimize_qed_evaluate.py \
    --restore-dir $1 \
    --checkpoint $2 \
    --min-atoms 6 \
    --max-atoms 25 \
    --max-num-bonds 100
