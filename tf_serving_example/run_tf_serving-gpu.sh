#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container

SINGULARITYENV_MODEL_NAME='ysi_model' singularity exec --nv -B ./ysi_model:/models/ysi_model -B ./batch.config:/models/batch.config /projects/rlmolecule/pstjohn/containers/tensorflow-serving-gpu.simg tf_serving_entrypoint.sh --enable_batching --batching_parameters_file='/models/batch.config'

