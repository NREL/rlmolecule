import os
import logging
import sys

import numpy as np
import random

import rdkit
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcNumUnspecifiedAtomStereoCenters
from rdkit.Chem.rdDistGeom import EmbedMolecule

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from alphazero.molecule_filters import check_all_filters

pt = Chem.GetPeriodicTable()

bond_orders = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

def get_free_valence(atom):
    """ For a given atom, calculate the free valence remaining """
    return pt.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()


def build_molecules(starting_mol,
                    atom_additions=None,
                    stereoisomers=True,
                    sa_score_threshold=3.,
                    tryEmbedding=True,
                    max_energy_diff=15.):
    
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
            
        max_energy_diff: Optional[float]
            maximum MMFF between starting and final molecule
            
    Yields:
        rdkit.Mol, corresponding to modified input
    
    """
    
    if stereoisomers:
        # If we are building stereoisomers, we shouldn't have
        # unassigned stereochem to begin with
        if CalcNumUnspecifiedAtomStereoCenters(starting_mol) != 0:
            logging.warn(f'{rdkit.Chem.MolToSmiles(starting_mol)} has undefined stereochemistry')
        rdkit.Chem.FindPotentialStereoBonds(starting_mol)
        for bond in starting_mol.GetBonds():
            if bond.GetStereo() == rdkit.Chem.rdchem.BondStereo.STEREOANY:
                logging.warn(f'{rdkit.Chem.MolToSmiles(starting_mol)} has undefined stereo bonds')
    
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

    generated_smiles = []
    
#     if max_energy_diff is not None:
#         starting_energy = get_minimum_ff_energy(starting_mol)
    
    # Construct the generator
    for i, atom in enumerate(starting_mol.GetAtoms()):
        for partner in get_valid_partners(atom):
            for bond_order in get_valid_bonds(i, partner):
                mol = add_bond(i, partner, bond_order)

                # Not ideal to roundtrip SMILES here, but there's cleaning steps that often get missed if we dont
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#                 Chem.SanitizeMol(mol)
#                 rdkit.Chem.rdmolops.AssignStereochemistry(mol, flagPossibleStereoCenters=True)

                if not check_all_filters(mol):
                    continue
                
                for isomer in enumerate_stereoisomers(mol, stereoisomers):
                    smiles = Chem.MolToSmiles(isomer)
                    logging.debug(smiles)
                    if smiles not in generated_smiles:
                        generated_smiles += [smiles]
                        
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
                                
                        yield Chem.MolFromSmiles(smiles)

                        
def build_radicals(starting_mol):
    """This is a bit application-specific, as we're looking to build 
    organic radicals. """
    
    generated_smiles = []
    
    for i, atom in enumerate(starting_mol.GetAtoms()):
        if get_free_valence(atom) > 0:
            rw_mol = rdkit.Chem.RWMol(starting_mol)
            rw_mol.GetAtomWithIdx(i).SetNumRadicalElectrons(1)
            
            Chem.SanitizeMol(rw_mol)            
            smiles = Chem.MolToSmiles(rw_mol)
            if smiles not in generated_smiles:
                 # This makes sure the atom ordering is standardized
                yield Chem.MolFromSmiles(smiles) 
                generated_smiles += [smiles]
                