#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container/3.6.1

SINGULARITYENV_MODEL_NAME=ysi_model singularity exec -B ./ysi_model:/models/ysi_model /projects/rlmolecule/pstjohn/containers/tensorflow-serving.simg tf_serving_entrypoint.sh

