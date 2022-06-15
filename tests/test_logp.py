import numpy as np
import rdkit
from rlmolecule.examples.logp import penalized_logp


def test_penalized_logp():
    assert np.isclose(penalized_logp(rdkit.Chem.MolFromSmiles("CCCCCC")), 1.37756945)
