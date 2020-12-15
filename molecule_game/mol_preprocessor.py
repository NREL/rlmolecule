import os

import nfp
import numpy as np
import rdkit


def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
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


def bond_featurizer(bond, flipped=False):
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


def filter_keys(attribute):
    return {key: value for key, value in attribute.items()
            if key in {'atom', 'bond', 'connectivity'}}


class MolPreprocessor(nfp.preprocessing.SmilesPreprocessor):
    output_types = filter_keys(nfp.preprocessing.SmilesPreprocessor.output_types)
    output_shapes = filter_keys(nfp.preprocessing.SmilesPreprocessor.output_shapes)
    padding_values = filter_keys(nfp.preprocessing.SmilesPreprocessor.padding_values)
    
    def padded_shapes(self, *args, **kwargs):
        return filter_keys(super().padded_shapes(*args, **kwargs))
    
    def construct_feature_matrices(self, mol, train=False):
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
            'atom':         atom_feature_matrix,
            'bond':         bond_feature_matrix,
            'connectivity': connectivity,
            }


preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                               bond_features=bond_featurizer,
                               explicit_hs=False)

preprocessor.from_json(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'preprocessor.json'))
