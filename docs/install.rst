============
Installation
============

RLMolecule can be installed with pip into a suitable python environment with

.. code-block:: bash

   pip install rlmolecule


The tensorflow and rdkit dependencies can be tricky to install. A recommended conda environment is

.. code-block:: yaml

    channels:
    - conda-forge
    - defaults
    dependencies:
    - python=3.7
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
    - pyyaml
    - pip
    - pip:
	- nfp
	- networkx
	- tensorflow-gpu==2.3.0
	- tensorflow-addons
