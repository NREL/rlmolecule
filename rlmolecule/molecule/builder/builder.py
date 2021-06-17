import logging
import os
import pathlib
import sys
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Iterable, List, Optional

import numpy as np
import rdkit
from joblib import Memory
from rdkit import Chem, RDConfig
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdDistGeom import EmbedMolecule

from rlmolecule.molecule.builder.gdb_filters import check_all_filters

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# noinspection PyUnresolvedReferences
import sascorer

pt = Chem.GetPeriodicTable()
bond_orders = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
logger = logging.getLogger(__name__)


class MoleculeBuilder:
    def __init__(self,
                 max_atoms: int = 10,
                 min_atoms: int = 4,
                 atom_additions: Optional[List] = None,
                 stereoisomers: bool = True,
                 sa_score_threshold: Optional[float] = 3.5,
                 tryEmbedding: bool = True,
                 cache_dir: Optional[pathlib.Path] = None) -> None:
        """A class to build molecules according to a number of different options

        :param max_atoms: Maximum number of heavy atoms
        :param min_atoms: minimum number of heavy atoms
        :param atom_additions: potential atom types to consider. Defaults to ('C', 'H', 'O')
        :param stereoisomers: whether to consider stereoisomers different molecules
        :param sa_score_threshold: If set, don't construct molecules greater than a given sa_score.
        :param tryEmbedding: Try to get a 3D embedding of the molecule, and if this fails, remote it.
        """

        # Not the most elegant solution, these are carried and referenced by MoleculeState, but are not used internally
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms

        if cache_dir is not None:
            self.mem = Memory(cachedir=cache_dir)
            self.cached_call = self.mem.cache(self.call_list)

        else:
            self.mem = None
            self.cached_call = None


        self.transformation_stack = [
            AddNewAtomsAndBonds(atom_additions),
            GdbFilter(),
        ]

        if sa_score_threshold is not None:
            self.transformation_stack += [SAScoreFilter(sa_score_threshold, min_atoms)]

        if stereoisomers:
            self.transformation_stack += [StereoEnumerator()]

        if tryEmbedding:
            self.transformation_stack += [EmbeddingFilter()]

    def call_iter(self, parent_molecule: rdkit.Chem.Mol) -> Iterable[rdkit.Chem.Mol]:
        inputs = [parent_molecule]
        for transformer in self.transformation_stack:
            inputs = transformer(inputs)
        yield from inputs

    def call_list(self, parent_molecule: rdkit.Chem.Mol) -> List[rdkit.Chem.Mol]:
        return list(self.call_iter(parent_molecule))

    def __call__(self, parent_molecule: rdkit.Chem.Mol) -> Iterable[rdkit.Chem.Mol]:
        if self.cached_call is None:
            return self.call_iter(parent_molecule)

        else:
            return self.cached_call(parent_molecule)


class MoleculeTransformer(ABC):
    def __init__(self, max_threads: Optional[int] = 1):
        self.max_threads = max_threads

    @abstractmethod
    def call(self, molecule: rdkit.Chem.Mol) -> Iterable[rdkit.Chem.Mol]:
        pass

    def call_list(self, molecule: rdkit.Chem.Mol) -> List[rdkit.Chem.Mol]:
        return list(self.call(molecule))

    def __call__(self, inputs: Iterable[rdkit.Chem.Mol]) -> Iterable[rdkit.Chem.Mol]:
        if self.max_threads == 1:
            for molecule in inputs:
                yield from self.call(molecule)

        else:
            with Pool(self.max_threads) as p:
                for result in p.map(self.call_list, list(inputs)):
                    yield from result


class MoleculeFilter(MoleculeTransformer, ABC):
    @abstractmethod
    def filter(self, molecule: rdkit.Chem.Mol) -> bool:
        pass

    def call(self, molecule: rdkit.Chem.Mol) -> Iterable[rdkit.Chem.Mol]:
        if self.filter(molecule):
            yield molecule


class UniqueMoleculeTransformer(MoleculeTransformer, ABC):
    def __call__(self, inputs: Iterable[rdkit.Chem.Mol]):
        generated_smiles = set()
        for mol in super(UniqueMoleculeTransformer, self).__call__(inputs):
            smiles = Chem.MolToSmiles(mol)
            if smiles not in generated_smiles:
                generated_smiles.add(smiles)
                yield Chem.MolFromSmiles(smiles)


class AddNewAtomsAndBonds(UniqueMoleculeTransformer):
    def __init__(self, atom_additions: Optional[List] = None, **kwargs):
        super(AddNewAtomsAndBonds, self).__init__(**kwargs)
        if atom_additions is not None:
            self.atom_additions = atom_additions
        else:
            self.atom_additions = ('C', 'N', 'O')

    def call(self, molecule) -> Iterable[rdkit.Chem.Mol]:
        for i, atom in enumerate(molecule.GetAtoms()):
            for partner in self._get_valid_partners(molecule, atom):
                for bond_order in self._get_valid_bonds(molecule, i, partner):
                    yield self._add_bond(molecule, i, partner, bond_order)

    @staticmethod
    def _get_free_valence(atom) -> int:
        """ For a given atom, calculate the free valence remaining """
        return pt.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()

    def _get_valid_partners(self, starting_mol: rdkit.Chem.Mol, atom: rdkit.Chem.Atom) -> List[int]:
        """ For a given atom, return other atoms it can be connected to """
        return list(
            set(range(starting_mol.GetNumAtoms())) - set((neighbor.GetIdx() for neighbor in atom.GetNeighbors())) -
            set(range(atom.GetIdx())) -  # Prevent duplicates by only bonding forward
            set((atom.GetIdx(),)) | set(np.arange(len(self.atom_additions)) + starting_mol.GetNumAtoms()))

    def _get_valid_bonds(self, starting_mol: rdkit.Chem.Mol, atom1_idx: int, atom2_idx: int) -> range:
        """ Compare free valences of two atoms to calculate valid bonds """
        free_valence_1 = self._get_free_valence(starting_mol.GetAtomWithIdx(atom1_idx))
        if atom2_idx < starting_mol.GetNumAtoms():
            free_valence_2 = self._get_free_valence(starting_mol.GetAtomWithIdx(int(atom2_idx)))
        else:
            free_valence_2 = pt.GetDefaultValence(self.atom_additions[atom2_idx - starting_mol.GetNumAtoms()])

        return range(min(min(free_valence_1, free_valence_2), 3))

    def _add_bond(self, starting_mol: rdkit.Chem.Mol, atom1_idx: int, atom2_idx: int, bond_type: int) -> Chem.RWMol:
        """ Given two atoms and a bond type, execute the addition using rdkit """
        num_atom = starting_mol.GetNumAtoms()
        rw_mol = Chem.RWMol(starting_mol)

        if atom2_idx < num_atom:
            rw_mol.AddBond(atom1_idx, atom2_idx, bond_orders[bond_type])

        else:
            rw_mol.AddAtom(Chem.Atom(self.atom_additions[atom2_idx - num_atom]))
            rw_mol.AddBond(atom1_idx, num_atom, bond_orders[bond_type])

        return rw_mol


class StereoEnumerator(UniqueMoleculeTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opts = StereoEnumerationOptions(unique=True)

    def call(self, molecule: rdkit.Chem.Mol) -> Iterable[rdkit.Chem.Mol]:
        return EnumerateStereoisomers(molecule, options=self.opts)


class SAScoreFilter(MoleculeFilter):
    def __init__(self, sa_score_threshold: float, min_atoms: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.min_atoms = min_atoms
        self.sa_score_threshold = sa_score_threshold

    def filter(self, molecule: rdkit.Chem.Mol) -> bool:
        if molecule.GetNumAtoms() >= self.min_atoms:
            return sascorer.calculateScore(molecule) <= self.sa_score_threshold
        return True


class EmbeddingFilter(MoleculeFilter):
    def filter(self, molecule: rdkit.Chem.Mol) -> bool:
        molH = Chem.AddHs(molecule)

        try:
            assert EmbedMolecule(molH) >= 0
            return True

        except (AssertionError, RuntimeError):
            return False


class GdbFilter(MoleculeFilter):
    def filter(self, molecule: rdkit.Chem.Mol) -> bool:
        try:
            return check_all_filters(molecule)
        except Exception as ex:
            logger.warning(f"Issue with GDBFilter and molecule {Chem.MolToSmiles(molecule)}: {ex}")
            return False
