#!/bin/bash
module purge
source activate /home/dbiagion/conda-envs/rlmol
export PYTHONPATH=$PYTHONPATH:$PWD/command_line_tools
unset LD_PRELOAD
module load cudnn/8.1.1
