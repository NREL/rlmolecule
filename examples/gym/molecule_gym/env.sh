#!/bin/bash
module purge
# conda activate /home/dbiagion/conda-envs/rlmol
conda activate rlmol
export PYTHONPATH=$PYTHONPATH:$PWD/command_line_tools
unset LD_PRELOAD
module load cudnn/8.1.1/cuda-11.2
