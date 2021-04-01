# rlmolecule

## About

A library for general-purpose material and molecular optimization using AlphaZero-style reinforcement learning.

Code currently under development as part of the ["End-to-End Optimization for Battery Materials and Molecules by Combining Graph Neural Networks and Reinforcement Learning"](https://arpa-e.energy.gov/technologies/projects/end-end-optimization-battery-materials-and-molecules-combining-graph-neural) project at the National Renewable Energy Laboratory (NREL), Colorado School of Mines (CSU), and Colorado State University (CSU). Funding provided by the Advanced Research Projects Agency–Energy (ARPA-E)'s [DIFFERENTIATE program](https://arpa-e.energy.gov/technologies/programs/differentiate)


## Installation

Most dependencies for this project are installable via conda, with the exception of [nfp](https://github.com/NREL/nfp), which can be installed via pip. An example conda environment file is provided below:

```yaml
name: tf2_gpu
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - xtb-python
  - ase
  - pytorch-cpu
  - jupyterlab
  - rdkit
  - seaborn
  - pandas
  - scikit-learn
  - jupyter
  - notebook
  - pymatgen
  - xlrd
  - tqdm
  - psycopg2
  - cudnn=7.6
  - sqlalchemy
  - pip
  - pip:
    - nfp
    - networkx
    - tensorflow-gpu==2.3.0
    - tensorflow-addons
```


With the above file as `environment.yml`, you can install on Eagle with 
```bash
module purge
conda env create -f environment.yml --prefix /projects/rlmolecule/<USER>/envs/tf2_gpu
```

## Usage

This library is still under active development, and APIs are expected to change frequently. Currently, optimization proceeds by subclassing `alphazero.Node` to provide the expected reward function calculation. Example usage of the module is demonstrated for radical construction in `stable_radical_optimization/`.
