from typing import Iterable, List, Optional

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolToSmiles,
)

from rlmolecule.molecule.molecule_tools import build_molecules
from rlmolecule.state import State


class MoleculeConfig:
    def __init__(self,
                 max_atoms: int = 10,
                 min_atoms: int = 4,
                 atom_additions: Optional[List] = None,
                 stereoisomers: bool = True,
                 sa_score_threshold: Optional[float] = 3.,
                 tryEmbedding: bool = True) -> None:
        """A configuration class to contain a number of different molecule construction parameters.

        :param max_atoms: Maximum number of heavy atoms
        :param min_atoms: minimum number of heavy atoms
        :param atom_additions: potential atom types to consider. Defaults to ('C', 'H', 'O')
        :param stereoisomers: whether to consider stereoisomers different molecules
        :param sa_score_threshold: If set, don't construct molecules greater than a given sa_score.
        :param tryEmbedding: Try to get a 3D embedding of the molecule, and if this fails, remote it.
        """
        self.max_atoms = max_atoms
        self.stereoisomers = stereoisomers
        self.min_atoms = min_atoms
        self.atom_additions = atom_additions
        self.sa_score_threshold = sa_score_threshold
        self.tryEmbedding = tryEmbedding


class MoleculeState(State):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """

    def __init__(self, molecule: Mol, config: Optional[MoleculeConfig] = None, is_terminal:bool = False) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param config: A MoleculeConfig class
        :param is_terminal: Whether to consider this molecule as a final state
        """
        if config == None:
            config = MoleculeConfig()
        self._config = config
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule)
        self._is_terminal = is_terminal

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return self._smiles

    def equals(self, other: 'MoleculeState') -> bool:
        """
        delegates to the SMILES string
        """
        return isinstance(other, MoleculeState) and self._smiles == other._smiles

    def hash(self) -> int:
        """
        delegates to the SMILES string
        """
        return hash(self._smiles)

    @property
    def config(self) -> MoleculeConfig:
        return self._config

    @property
    def smiles(self) -> str:
        return self._smiles

    @property
    def molecule(self) -> Mol:
        return self._molecule

    @property
    def num_atoms(self) -> int:
        return self.molecule.GetNumAtoms()

    def get_next_actions(self) -> Iterable['State']:
        if (self.num_atoms < self.config.max_atoms) and not self._is_terminal:
            bml = list((self.__class__(molecule, self.config) for molecule in
                        build_molecules(
                            self.molecule,
                            atom_additions=self.config.atom_additions,
                            stereoisomers=self.config.stereoisomers,
                            sa_score_threshold=self.config.sa_score_threshold,
                            tryEmbedding=self.config.tryEmbedding
                        )))
            yield from bml

        if (self.num_atoms >= self.config.min_atoms) and not self._is_terminal:
            yield self.__class__(self.molecule, self.config, is_terminal=True)
