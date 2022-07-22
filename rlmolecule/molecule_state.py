# TODO:
# * clean up action caching
# * clean up CSV reward logging

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import gym
import nfp
import numpy as np
import ray
from graphenv.vertex import V, Vertex
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from rlmolecule.actors import get_csv_logger, get_terminal_cache
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.policy.preprocessor import load_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class MoleculeData:
    builder: Type[MoleculeBuilder]
    max_num_actions: int = 20
    max_num_bonds: Optional[int] = None
    preprocessor: Union[Type[nfp.preprocessing.MolPreprocessor], str, None] = None
    prune_terminal_states: bool = False
    terminal_cache: Optional[Any] = None
    using_ray: Optional[bool] = None
    log_reward_filepath: Optional[str] = None

    def __post_init__(self):
        if self.max_num_bonds is None:
            self.max_num_bonds = self.builder.max_atoms * 4

        if self.preprocessor is None or isinstance(self.preprocessor, str):
            self.preprocessor = load_preprocessor(self.preprocessor)

        if self.prune_terminal_states and self.terminal_cache is None:
            if ray.is_initialized():
                self.terminal_cache = get_terminal_cache()
                self.using_ray = True
            else:
                self.terminal_cache = set()
                self.using_ray = False

        self.csv_writer = None
        if ray.is_initialized() and self.log_reward_filepath is not None:
            self.csv_writer = get_csv_logger(self.log_reward_filepath)

    def log_reward(self, row: List):
        logger.info(f"REWARD: {row}")
        if self.csv_writer is not None:
            self.csv_writer.write.remote(row)


class MoleculeState(Vertex):
    def __init__(
        self,
        molecule: Mol,
        data: Type[MoleculeData],
        smiles: Optional[str] = None,
        force_terminal: bool = False,
    ) -> None:
        """
        A state implementation which uses simple transformations (such as adding a bond)
        to define a graph of molecules that can be navigated.

        Molecules are stored as rdkit Mol instances, and the rdkit-generated SMILES
        string is also stored for efficient hashing.

        :param molecule: an RDKit molecule specifying the current state
        :param builder: A MoleculeConfig class
        :param force_terminal: Whether to force this molecule to be a terminal state
        :param smiles: An optional smiles string for the molecule; must match
        `molecule`.
        """
        super().__init__()
        self._molecule: Mol = molecule
        self._smiles: str = MolToSmiles(self._molecule) if smiles is None else smiles
        self._forced_terminal: bool = force_terminal
        self.data = data

    @property
    def root(self) -> V:
        return self.new(MolFromSmiles("C"))

    def _get_children(self) -> Sequence[V]:

        if self.forced_terminal:
            # No children from a molecule that's flagged as a terminal state, this
            # makes the Vertex.terminal call evaluate to True
            return []

        next_actions = [self.new(molecule) for molecule in self.builder(self.molecule)]
        next_actions.extend(self._get_terminal_actions())

        if self.data.prune_terminal_states:
            next_actions = self._prune_next_actions(next_actions)

        if len(next_actions) >= self.max_num_actions:
            logger.info(
                f"{self} has {len(next_actions) + 1} next actions when the "
                f"maximum is {self.max_num_actions}"
            )
            next_actions = random.sample(next_actions, self.max_num_actions)

        return next_actions

    def _get_terminal_actions(self) -> Sequence[V]:
        return [self.new(self.molecule, force_terminal=True, smiles=self.smiles)]

    def _prune_next_actions(self, next_actions: Sequence[V]):
        smiles_list = [repr(mol) for mol in next_actions]
        if self.data.using_ray:
            to_prune = ray.get(self.data.terminal_cache.contains.remote(smiles_list))
        else:
            to_prune = [smiles in self.data.terminal_cache for smiles in smiles_list]

        next_actions = [
            mol for mol, contained in zip(next_actions, to_prune) if not contained
        ]

        return next_actions

    def _make_observation(self) -> Dict[str, np.ndarray]:
        return self.preprocessor(
            self.molecule,
            max_num_nodes=self.builder.max_atoms,
            max_num_edges=self.data.max_num_bonds,
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
                    shape=(self.data.max_num_bonds,),
                    dtype=int,
                ),
                "connectivity": gym.spaces.Box(
                    low=0,
                    high=self.builder.max_atoms,
                    shape=(self.data.max_num_bonds, 2),
                    dtype=int,
                ),
            }
        )

    def new(
        self,
        molecule: Mol,
        *args,
        smiles: Optional[str] = None,
        force_terminal: bool = False,
        **kwargs,
    ) -> V:
        new = self.__class__(
            molecule,
            self.data,
            smiles=smiles,
            force_terminal=force_terminal,
            *args,
            **kwargs,
        )
        return new

    @property
    def forced_terminal(self) -> bool:
        return self._forced_terminal

    @property
    def builder(self) -> any:
        return self.data.builder

    @property
    def preprocessor(self) -> any:
        return self.data.preprocessor

    @property
    def smiles(self) -> str:
        return self._smiles

    @property
    def molecule(self) -> Mol:
        return self._molecule

    @property
    def num_atoms(self) -> int:
        return self.molecule.GetNumAtoms()

    @property
    def max_num_actions(self) -> int:
        return self.data.max_num_actions

    def __repr__(self) -> str:
        """
        delegates to the SMILES string
        """
        return f"{self.smiles}{' (t)' if self._forced_terminal else ''}"

    @property
    def terminal(self) -> bool:
        is_terminal = super().terminal
        if is_terminal and self.data.prune_terminal_states:
            if self.data.using_ray:
                self.data.terminal_cache.add.remote(repr(self))
            else:
                self.data.terminal_cache.add(repr(self))

        return is_terminal
