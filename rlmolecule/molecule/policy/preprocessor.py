import os
from typing import Dict, Optional

import nfp
import numpy as np
import rdkit


def atom_featurizer(atom: rdkit.Chem.Atom) -> str:
    """ Return an string representing the atom type
    :param atom: the rdkit.Atom object  
    :return: a string representation for embedding
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag().name,
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond: rdkit.Chem.Bond, flipped: bool = False) -> str:
    """Return a string representation of the given bond

    :param bond: The rdkit bond object
    :param flipped: Whether the bond is considered in the forward or reverse direction
    :return: a string representation of the bond type
    """
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))

    bstereo = bond.GetStereo().name
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''

    return " ".join([atoms, btype, ring, bstereo]).strip()


def filter_keys(attribute: Dict) -> Dict:
    """Remove unnecessary model inputs from nfp.SmilesPreprocessor outputs

    :param attribute: A dictionary containing unnecessary keys
    :return: The same dictionary with only 'atom', 'bond', and 'connectivity' arrays
    """
    return {key: value for key, value in attribute.items()
            if key in {'atom', 'bond', 'connectivity'}}


class MolPreprocessor(nfp.preprocessing.SmilesPreprocessor):
    output_types = filter_keys(nfp.preprocessing.SmilesPreprocessor.output_types)
    output_shapes = filter_keys(nfp.preprocessing.SmilesPreprocessor.output_shapes)
    padding_values = filter_keys(nfp.preprocessing.SmilesPreprocessor.padding_values)

    def padded_shapes(self, *args, **kwargs):
        return filter_keys(super().padded_shapes(*args, **kwargs))

    def construct_feature_matrices(self, mol: rdkit.Chem.Mol, train: bool = False) -> {}:
        """ Convert an rdkit Mol to a list of tensors
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.
        """

        self.atom_tokenizer.train = train
        self.bond_tokenizer.train = train

        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)

        n_atom = mol.GetNumAtoms()
        n_bond = 2 * mol.GetNumBonds()

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1

        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        if n_bond == 1:
            bond_feature_matrix[0] = self.bond_tokenizer('self-link')

        bond_index = 0
        for n, atom in enumerate(mol.GetAtoms()):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[bond_index] = self.bond_tokenizer(
                    self.bond_features(bond, flipped=rev))

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1

        return {
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }


def load_preprocessor(saved_preprocessor_file: Optional[str] = None) -> MolPreprocessor:
    """Load the MolPreprocessor object from either the default json file or a provided data file

    :param saved_preprocessor_file: directory of the saved nfp.Preprocessor json data
    :return: a MolPreprocessor instance for the molecule policy network
    """
    preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                                   bond_features=bond_featurizer,
                                   explicit_hs=False)

    if not saved_preprocessor_file:
        saved_preprocessor_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessor.json')

    preprocessor.from_json(saved_preprocessor_file)
    return preprocessor
