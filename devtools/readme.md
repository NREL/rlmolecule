# Conda environments for development and testing

The GPU environment uses tensorflow installed via pip, which for 2.4.1 requires CUDA 11 and CUDNN 8.0

* https://github.com/NREL/HPC/tree/master/workshops/Optimized_TF
* https://www.tensorflow.org/install/source#gpu

Installing on eagle:

```bash
module purge
module load conda
module load cudnn/8.1.1/cuda-11.2

conda env create -f ./conda-envs/dev.yaml
source activate rlmol
```
