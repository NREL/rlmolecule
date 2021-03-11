import os
import random
import sys

import numpy as np
import rdkit
from rlmolecule.molecule.molecule_filters import check_all_filters
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.rdDistGeom import EmbedMolecule

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

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


def build_molecules(starting_mol,
                    atom_additions=None,
                    stereoisomers=True,
                    sa_score_threshold=3.,
                    tryEmbedding=True):
    """Return an iterator of molecules that result from a single manipulation 
    (i.e., atom / bond addition) to the starting molecule
    
    Arguments:
        starting_mol: rdkit.Mol
            The starting molecule as an rdkit Mol object
        
        atom_additions: list of elements
            Types of atoms that can be added. Defaults to ('C', 'N', 'O')
            
        stereoisomers: bool
            Whether to iterate over potential stereoisomers of the given molecule
            as seperate molecules
            
        sa_score_threshold: float or None
            Whether to calculate the sa_score of the given molecule, and withold 
            molecules that have a sa_score higher than the threshold.
            
        tryEmbedding: bool
            whether to try an rdkit 3D embedding of the molecule
            
    Yields:
        rdkit.Mol, corresponding to modified input
    
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

    def enumerate_stereoisomers(mol, to_use):
        """ We likely want to distinguish between stereoisomers, so we do that here """

        if not to_use:
            # Give an easy way to pass through this function if this feature isn't used
            return (mol,)

        else:
            opts = StereoEnumerationOptions(unique=True)
            return tuple(EnumerateStereoisomers(mol, options=opts))

    generated_smiles = {}

    # Construct the generator
    for i, atom in enumerate(starting_mol.GetAtoms()):
        for partner in get_valid_partners(atom):
            for bond_order in get_valid_bonds(i, partner):
                mol = add_bond(i, partner, bond_order)

                # Not ideal to roundtrip SMILES here, but there's cleaning steps that often get
                # missed if we just santize
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

                if not check_all_filters(mol):
                    continue

                for isomer in enumerate_stereoisomers(mol, stereoisomers):
                    smiles = Chem.MolToSmiles(mol)
                    if smiles not in generated_smiles:
                        generated_smiles[smiles] = True

                        if sa_score_threshold is not None:
                            if sascorer.calculateScore(isomer) >= sa_score_threshold:
                                continue

                        if tryEmbedding:
                            ntm = Chem.AddHs(isomer)

                            try:
                                assert EmbedMolecule(ntm) >= 0
                            except (AssertionError, RuntimeError):
                                # Failed a 3D embedding                                
                                continue

                        yield rdkit.Chem.MolFromSmiles(smiles)


def build_radicals(starting_mol):
    """This is a bit application-specific, as we're looking to build 
    organic radicals. """

    generated_smiles = {}

    for i, atom in enumerate(starting_mol.GetAtoms()):
        if get_free_valence(atom) > 0:
            rw_mol = rdkit.Chem.RWMol(starting_mol)
            rw_mol.GetAtomWithIdx(i).SetNumRadicalElectrons(1)

            Chem.SanitizeMol(rw_mol)
            smiles = Chem.MolToSmiles(rw_mol)
            if smiles not in generated_smiles:
                # This makes sure the atom ordering is standardized
                yield Chem.MolFromSmiles(smiles)
                generated_smiles[smiles] = True
