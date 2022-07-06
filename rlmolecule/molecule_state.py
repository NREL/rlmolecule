import logging
import random
from typing import Dict, Optional, Sequence, Type, Union

import gym
import nfp
import numpy as np
from graphenv.vertex import V, Vertex
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from rlmolecule.builder import MoleculeBuilder
from rlmolecule.policy.preprocessor import load_preprocessor

logger = logging.getLogger(__name__)


class MoleculeState(Vertex):
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
        max_num_bonds: Optional[int] = None,
        preprocessor: Union[Type[nfp.preprocessing.MolPreprocessor], str, None] = None,
        warn: bool = True,
    ) -> None:
        """
        :param molecule: an RDKit molecule specifying the current state
        :param builder: A MoleculeConfig class
        :param force_terminal: Whether to force this molecule to be a terminal state
        :param smiles: An optional smiles string for the molecule; must match
        `molecule`.
        :param max_num_actions: The maximum number of next states to consider.
        :param warn: whether to warn if more than the max_num_actions are possible.
        """
        super().__init__()
        self._builder: any = builder
        self.max_num_bonds = (
            builder.max_atoms * 4 if max_num_bonds is None else max_num_bonds
        )
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule) if smiles is None else smiles
        self._forced_terminal: bool = force_terminal
        self.max_num_actions = max_num_actions
        self._warn = warn

        if preprocessor is None or isinstance(preprocessor, str):
            self.preprocessor = load_preprocessor(preprocessor)
        else:
            self.preprocessor = preprocessor

    @property
    def root(self) -> V:
        return self.new(MolFromSmiles("C"))

    def _get_children(self) -> Sequence[V]:
        """TODO: should have an option not to yield terminal states that have already
        been explored. That would require us to cache terminal states (both
        `forced_terminal` and those without children) and prune those from the search
        tree before down-selecting to max_num_actions.
        """

        if self.forced_terminal:
            return []

        next_molecules = list(self.builder(self.molecule))
        if len(next_molecules) >= self.max_num_actions:
            if self._warn:
                logger.warning(
                    f"{self} has {len(next_molecules) + 1} next actions when the "
                    f"maximum is {self.max_num_actions}"
                )
            next_molecules = random.sample(next_molecules, self.max_num_actions - 1)

        next_actions = [self.new(molecule) for molecule in next_molecules]
        next_actions.append(
            self.new(self.molecule, force_terminal=True, smiles=self.smiles)
        )

        logger.debug(f"Returning {len(next_actions)} for state {self.smiles}")

        return next_actions

    def _make_observation(self) -> Dict[str, np.ndarray]:
        return self.preprocessor(
            self.molecule,
            max_num_nodes=self.builder.max_atoms,
            max_num_edges=self.max_num_bonds,
        )

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "atom": gym.spaces.Box(
                    low=0,
                    high=self.preprocessor.atom_classes,
                    shape=(self.builder.max_atoms,),
                    dtype=int,
                ),
                "bond": gym.spaces.Box(
                    low=0,
                    high=self.preprocessor.bond_classes,
                    shape=(self.max_num_bonds,),
                    dtype=int,
                ),
                "connectivity": gym.spaces.Box(
                    low=0,
                    high=self.builder.max_atoms,
                    shape=(self.max_num_bonds, 2),
                    dtype=int,
                ),
            }
        )

    def new(
        self, molecule: Mol, force_terminal: bool = False, smiles: Optional[str] = None,
    ) -> V:
        return self.__class__(
            molecule,
            self.builder,
            force_terminal,
            smiles,
            max_num_actions=self.max_num_actions,
            max_num_bonds=self.max_num_bonds,
            preprocessor=self.preprocessor,
            warn=self._warn,
        )

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

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return f"{self.smiles}{' (t)' if self._forced_terminal else ''}"
