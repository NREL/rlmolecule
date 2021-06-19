import sys

import numpy as np
import pandas as pd
import rdkit
import tensorflow as tf
from rdkit import Chem

sys.path.append('/projects/rlmolecule/pstjohn/models/20201031_bde/')
from preprocess_inputs import preprocessor as bde_preprocessor

bde_preprocessor.from_json('/projects/rlmolecule/pstjohn/models/20201031_bde/preprocessor.json')


def prepare_for_bde(mol: rdkit.Chem.Mol) -> pd.Series:
    radical_index = None
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetNumRadicalElectrons() != 0:
            assert radical_index is None
            radical_index = i

            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            atom.SetNumRadicalElectrons(0)
            break
    else:
        raise RuntimeError(f"No radical found: {Chem.MolToSmiles(mol)}")

    radical_rank = Chem.CanonicalRankAtoms(mol, includeChirality=True)[radical_index]

    mol_smiles = Chem.MolToSmiles(mol)
    # TODO this line seems redundant
    mol = Chem.MolFromSmiles(mol_smiles)

    radical_index_reordered = list(Chem.CanonicalRankAtoms(mol, includeChirality=True)).index(radical_rank)

    molH = Chem.AddHs(mol)
    for bond in molH.GetAtomWithIdx(radical_index_reordered).GetBonds():
        if 'H' in {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}:
            bond_index = bond.GetIdx()
            break
    else:
        raise RuntimeError('Bond not found')

    h_bond_indices = [
        bond.GetIdx() for bond in filter(
            lambda bond: ((bond.GetEndAtom().GetSymbol() == 'H')
                          | (bond.GetBeginAtom().GetSymbol() == 'H')), molH.GetBonds())
    ]

    other_h_bonds = list(set(h_bond_indices) - {bond_index})

    return pd.Series({
        'mol_smiles': mol_smiles,
        'radical_index_mol': radical_index_reordered,
        'bond_index': bond_index,
        'other_h_bonds': other_h_bonds
    })


def bde_get_inputs(mol_smiles) -> {}:
    """ The BDE model was trained on a different set of data
    so we need to use corresponding preprocessor here
    """
    inputs = bde_preprocessor.construct_feature_matrices(mol_smiles, train=False)
    assert not (inputs['atom'] == 1).any() | (inputs['bond'] == 1).any()
    return {key: tf.constant(np.expand_dims(val, 0)) for key, val in inputs.items()}
