from typing import Optional, Sequence

from rdkit.Chem import Mol, MolToSmiles

from rlmolecule.molecule.molecule_building import build_molecules
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MoleculeState(GraphSearchState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """

    def __init__(self,
                 molecule: Mol,
                 config: any,
                 force_terminal: bool = False,
                 smiles: Optional[str] = None,
                 ) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param config: A MoleculeConfig class
        :param force_terminal: Whether to force this molecule to be a terminal state
        :param smiles: An optional smiles string for the molecule; must match `molecule`.
        """
        self._config: any = config
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule) if smiles is None else smiles
        self._forced_terminal: bool = force_terminal

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return f"{self._smiles}{' (t)' if self._forced_terminal else ''}"

    # noinspection PyUnresolvedReferences
    def equals(self, other: any) -> bool:
        """
        delegates to the SMILES string
        """
        return type(other) == type(self) and \
               self._smiles == other._smiles and \
               self._forced_terminal == other._forced_terminal

    def hash(self) -> int:
        """
        delegates to the SMILES string
        """
        return hash(self.__repr__()) ^ (13 * self._forced_terminal)

    def get_next_actions(self) -> Sequence['MoleculeState']:
        result = []
        if not self._forced_terminal:
            if self.num_atoms < self.config.max_atoms:
                result.extend((MoleculeState(molecule, self.config) for molecule in
                               build_molecules(
                                   self.molecule,
                                   atom_additions=self.config.atom_additions,
                                   stereoisomers=self.config.stereoisomers,
                                   sa_score_threshold=self.config.sa_score_threshold,
                                   tryEmbedding=self.config.tryEmbedding
                               )))

            if self.num_atoms >= self.config.min_atoms:
                result.append(MoleculeState(self.molecule, self.config, force_terminal=True))

        return result

    @property
    def forced_terminal(self) -> bool:
        return self._forced_terminal

    @property
    def config(self) -> any:
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
