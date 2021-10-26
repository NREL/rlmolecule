from typing import Optional

import gym
import numpy as np
import rdkit
# from nfp.preprocessing.mol_preprocessor import MolPreprocessor
from nfp.preprocessing import MolPreprocessor
from rdkit.Chem.QED import qed

from rlmolecule.graph_gym.graph_problem import GraphProblem
from rlmolecule.molecule.builder.builder import MoleculeBuilder
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.molecule.policy.preprocessor import load_preprocessor


class MoleculeGraphProblem(GraphProblem):

    def __init__(self,
                 builder: MoleculeBuilder,
                 preprocessor: Optional[MolPreprocessor] = None,
                 preprocessor_data: Optional[str] = None,
                 max_num_actions: int = 64,
                 max_num_bonds: int = 40,
                 ) -> None:
        super().__init__()
        self.builder: MoleculeBuilder = builder
        self.preprocessor: MolPreprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)
        self._max_num_actions: int = max_num_actions
        self.max_num_bonds: int = max_num_bonds

        # TODO: check ranges on these
        self._observation_space: gym.Space = gym.spaces.Dict({
            'atom': gym.spaces.Box(
                low=0, high=self.preprocessor.atom_classes,
                shape=(self.builder.max_atoms,), dtype=np.int),
            'bond': gym.spaces.Box(
                low=0, high=self.preprocessor.bond_classes,
                shape=(self.max_num_bonds,), dtype=np.int),
            'connectivity':
                gym.spaces.Box(
                    low=0, high=self.builder.max_atoms,
                    shape=(self.max_num_bonds, 2), dtype=np.int),
        })

        self._action_space: gym.Space = gym.spaces.Discrete(self.max_num_actions)

        self._null_observation = self._observation_space.sample()
        for v in self._null_observation.values():
            v *= 0

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def null_observation(self) -> any:
        return self._null_observation

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def max_num_actions(self) -> int:
        return self._max_num_actions

    def make_observation(self, state: MoleculeState) -> {str: np.ndarray}:
        return self.preprocessor.construct_feature_matrices(
            state.molecule,
            max_num_atoms=self.builder.max_atoms,
            max_num_bonds=self.max_num_bonds,
        )

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self.builder)

    def step(self, state: MoleculeState) -> (float, bool, dict):
        is_terminal = len(state.get_next_actions()) == 0
        if state.forced_terminal:
            r = qed(state.molecule)
            print(r)
            return r, is_terminal, {'forced_terminal': True, 'smiles': state.smiles}
            # return qed(state.molecule), is_terminal, {'forced_terminal': True, 'smiles': state.smiles}
        return 0.0, is_terminal, {'forced_terminal': False, 'smiles': state.smiles}

    @property
    def invalid_action_result(self) -> (float, bool, {}):
        return -1.0, True, {}
