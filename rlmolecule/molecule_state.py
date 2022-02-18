import logging
import random
from typing import Dict, Optional, Sequence, Type

import gym
import numpy as np
from graphenv.node import N, Node
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from rlmolecule.builder import MoleculeBuilder

logger = logging.getLogger(__name__)


class MoleculeState(Node):
    """
    A state implementation which uses simple transformations (such as adding a bond) to
    define a graph of molecules that can be navigated.

    Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES string
    is also stored for efficient hashing.
    """

    def __init__(
        self,
        molecule: Mol,
        builder: Type[MoleculeBuilder],
        force_terminal: bool = False,
        smiles: Optional[str] = None,
        max_num_actions: int = 20,
    ) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param builder: A MoleculeConfig class
        :param force_terminal: Whether to force this molecule to be a terminal state
        :param smiles: An optional smiles string for the molecule; must match
        `molecule`.
        :param max_num_actions: The maximum number of next states to consider.
        """
        super().__init__(max_num_actions)
        self._builder: any = builder
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule) if smiles is None else smiles
        self._forced_terminal: bool = force_terminal

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return f"{self.smiles}{' (t)' if self._forced_terminal else ''}"

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

    def get_next_actions(self) -> Sequence[N]:
        if self.forced_terminal:
            return []

        next_molecules = list(self.builder(self.molecule))
        if len(next_molecules) > self.max_num_actions - 1:
            logger.warning(
                f"{self} has {len(next_molecules) + 1} next actions when the "
                f"maximum is {self.max_num_actions}"
            )
            next_molecules = random.sample(next_molecules, self.max_num_actions)

        next_actions = [self.new(molecule) for molecule in next_molecules]
        next_actions.append(
            self.new(self.molecule, force_terminal=True, smiles=self.smiles)
        )
        return next_actions

    def make_observation(self) -> Dict[str, np.ndarray]:
        pass

    @property
    def observation_space(self) -> gym.spaces.Dict:
        pass

    def get_root(self) -> N:
        return self.new(MolFromSmiles("C"))

    def new(
        self, molecule: Mol, force_terminal: bool = False, smiles: Optional[str] = None
    ) -> N:
        return MoleculeState(molecule, self.builder, force_terminal, smiles)

    # @abstractmethod
    # def reward(self) -> float:
    #     pass
