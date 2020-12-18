from typing import Iterable

from rdkit.Chem import Mol, MolToSmiles

from molecule_game.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_tools import build_molecules
from rlmolecule.tree_search.tree_search_state import TreeSearchState


class MoleculeState(TreeSearchState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """

    def __init__(self, molecule: Mol, config: MoleculeConfig, is_terminal: bool = False) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param config: A MoleculeConfig class
        :param is_terminal: Whether to consider this molecule as a final state
        """
        self._config = config
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule)
        self._is_terminal = is_terminal

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return f"{self._smiles}{' (t)' if self.terminal else ''}"

    def equals(self, other: 'MoleculeState') -> bool:
        """
        delegates to the SMILES string
        """
        return (isinstance(other, self.__class__) and
                self._smiles == other._smiles and
                self.terminal == other.terminal)

    def hash(self) -> int:
        """
        delegates to the SMILES string
        """
        return hash(self.__repr__())

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

    def get_next_actions(self) -> Iterable['TreeSearchState']:
        if (self.num_atoms < self.config.max_atoms) and not self._is_terminal:
            bml = list((MoleculeState(molecule, self.config) for molecule in
                        build_molecules(
                            self.molecule,
                            atom_additions=self.config.atom_additions,
                            stereoisomers=self.config.stereoisomers,
                            sa_score_threshold=self.config.sa_score_threshold,
                            tryEmbedding=self.config.tryEmbedding
                        )))
            yield from bml

        if (self.num_atoms >= self.config.min_atoms) and not self._is_terminal:
            yield MoleculeState(self.molecule, self.config, is_terminal=True)
