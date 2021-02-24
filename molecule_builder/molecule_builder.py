import numpy as np
import random

import rdkit
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

pt = Chem.GetPeriodicTable()

bond_orders = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

def shuffle(item):
    """ For a given iterable, return a shuffled list """
    item_list = list(item)
    random.shuffle(item_list)
    return item_list

def get_free_valence(atom):
    """ For a given atom, calculate the free valence remaining """
    return pt.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()


def build_molecules(starting_mol, atom_additions=None, stereoisomers=False):
    """Return an iterator of molecules that result from a single manipulation 
    (i.e., atom / bond addition) to the starting molecule
    
    Arguments:
        starting_mol: rdkit.Mol
            The starting molecule as an rdkit Mol object
        
        atom_additions: list of elements
            Types of atoms that can be added. Defaults to ('C', 'N', 'O')
            
    Yields:
        rdkit.Mol, shuffled, corresponding to modified input
    
    """
    if atom_additions == None:
        atom_additions = ('C', 'N', 'O')

    def get_valid_partners(atom):
        """ For a given atom, return other atoms it can be connected to """
        return list(
            set(range(starting_mol.GetNumAtoms())) - 
            set((neighbor.GetIdx() for neighbor in atom.GetNeighbors())) -
            set(range(atom.GetIdx())) -  # Prevent duplicates by only bonding forward
            set((atom.GetIdx(),)) | 
            set(np.arange(len(atom_additions)) + starting_mol.GetNumAtoms()))

    def get_valid_bonds(atom1_idx, atom2_idx):
        """ Compare free valences of two atoms to calculate valid bonds """
        free_valence_1 = get_free_valence(starting_mol.GetAtomWithIdx(atom1_idx))
        if atom2_idx < starting_mol.GetNumAtoms():
            free_valence_2 = get_free_valence(starting_mol.GetAtomWithIdx(int(atom2_idx)))
        else:
            free_valence_2 = pt.GetDefaultValence(
                atom_additions[atom2_idx - starting_mol.GetNumAtoms()])

        return range(min(min(free_valence_1, free_valence_2), 3))

    def add_bond(atom1_idx, atom2_idx, bond_type):
        """ Given two atoms and a bond type, execute the addition using rdkit """
        num_atom = starting_mol.GetNumAtoms()
        rw_mol = Chem.RWMol(starting_mol)

        if atom2_idx < num_atom:
            rw_mol.AddBond(atom1_idx, atom2_idx, bond_orders[bond_type])

        else:
            rw_mol.AddAtom(Chem.Atom(
                atom_additions[atom2_idx - num_atom]))
            rw_mol.AddBond(atom1_idx, num_atom, bond_orders[bond_type])

        return rw_mol

    def enumerate_stereoisomers(mol):
        """ We likely want to distinguish between stereoisomers, so we do that here """
        opts = StereoEnumerationOptions(unique=True)
        return tuple(EnumerateStereoisomers(mol, options=opts))

    generated_smiles = []
    
    # Construct the generator
    for i, atom in shuffle(enumerate(starting_mol.GetAtoms())):
        for partner in shuffle(get_valid_partners(atom)):
            for bond_order in shuffle(get_valid_bonds(i, partner)):
                mol = add_bond(i, partner, bond_order)

                Chem.SanitizeMol(mol)
                if not stereoisomers:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles not in generated_smiles:
                        yield mol
                        generated_smiles += [smiles]
                    
                else:
                    for isomer in shuffle(enumerate_stereoisomers(mol)):
                        smiles = Chem.MolToSmiles(mol)
                        if smiles not in generated_smiles:                        
                            yield isomer
                            generated_smiles += [smiles]
                            