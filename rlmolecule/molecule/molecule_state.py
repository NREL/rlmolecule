from typing import Optional, Sequence, List

from rdkit.Chem import Mol, MolToSmiles

from rlmolecule.sql import hash_to_integer
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.metrics import collect_metrics


class MoleculeState(GraphSearchState):
    """
    A State implementation which uses simple transformations (such as adding a bond) to define a
    graph of molecules that can be navigated.
    
    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string is also stored for
    efficient hashing.
    """
    def __init__(
        self,
        molecule: Mol,
        builder: any,
        force_terminal: bool = False,
        smiles: Optional[str] = None,
    ) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param builder: A MoleculeConfig class
        :param force_terminal: Whether to force this molecule to be a terminal state
        :param smiles: An optional smiles string for the molecule; must match `molecule`.
        """
        self._builder: any = builder
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule) if smiles is None else smiles
        self._forced_terminal: bool = force_terminal
        self._next_states : Optional[List[MoleculeState]] = None

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
        return hash_to_integer(self.__repr__().encode())

    @collect_metrics
    def get_next_actions(self) -> Sequence['MoleculeState']:
        if self._next_states is None:
            next_states = []
            if not self._forced_terminal:
                if self.num_atoms < self.builder.max_atoms:
                    next_states.extend((MoleculeState(molecule, self.builder) for molecule in self.builder(self.molecule)))

                if self.num_atoms >= self.builder.min_atoms:
                    next_states.append(MoleculeState(self.molecule, self.builder, force_terminal=True))
            self._next_states = next_states

        return self._next_states

    @property
    def forced_terminal(self) -> bool:
        return self._forced_terminal

    @property
    def builder(self) -> any:
        return self._builder

    @property
    def smiles(self) -> str:
        return self._smiles

    @property
    def molecule(self) -> Mol:
        return self._molecule

    @property
    def num_atoms(self) -> int:
        return self.molecule.GetNumAtoms()
