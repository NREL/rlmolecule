#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container/3.2.1

SINGULARITYENV_MODEL_NAME=ysi_model singularity exec --nv -B ./ysi_model:/models/ysi_model /projects/rlmolecule/pstjohn/containers/tensorflow-serving-gpu.simg tf_serving_entrypoint.sh

