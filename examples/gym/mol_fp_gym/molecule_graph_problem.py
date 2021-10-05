from typing import Tuple, Dict

import gym
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.QED import qed
from sklearn.decomposition._pca import PCA

from rlmolecule.graph_gym.graph_problem import GraphProblem
from rlmolecule.molecule.builder.builder import MoleculeBuilder
from rlmolecule.molecule.molecule_state import MoleculeState

class MoleculeGraphProblem(GraphProblem):

    def __init__(self,
                 builder: MoleculeBuilder,
                 pca: PCA,
                 obs_bounds: Tuple[float, float] = (-1., 1.),
                 max_num_actions: int = 64,
                 max_num_bonds: int = 40,
                 ) -> None:
        super().__init__()
        self.builder: MoleculeBuilder = builder
        self.pca = pca
        self._max_num_actions: int = max_num_actions
        self.max_num_bonds: int = max_num_bonds

        self.pca_dim = pca.n_components_
        self.fp_size = pca.n_features_
        self.obs_bounds = obs_bounds

        # self._observation_space = gym.spaces.Dict({
        #     "fingerprint": gym.spaces.Box(
        #         low=self.obs_bounds[0], 
        #         high=self.obs_bounds[1],
        #         shape=(self.pca_dim,),
        #         dtype=np.float32)
        # })
        self._observation_space = gym.spaces.Box(
                low=self.obs_bounds[0], 
                high=self.obs_bounds[1],
                shape=(self.pca_dim,),
                dtype=np.float32
        )

        self._action_space: gym.Space = gym.spaces.Discrete(self.max_num_actions)

        self._null_observation = self._observation_space.sample()
        #for v in self._null_observation.values():
        for v in self._null_observation:
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

    def make_observation(self, state: MoleculeState) -> Dict[str, np.ndarray]:
        fp = np.array(Chem.RDKFingerprint(state.molecule, fpSize=self.fp_size))
        fp = fp.reshape(1, -1)
        fp = self.pca.transform(fp).squeeze()
        fp = np.clip(fp, *self.obs_bounds)
        #return {"fingerprint": fp.astype(np.float32)}
        return np.array(fp.astype(np.float32))

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self.builder)

    def step(self, state: MoleculeState) -> Tuple[float, bool, dict]:
        is_terminal = len(state.get_next_actions()) == 0
        if state.forced_terminal:
            return qed(state.molecule), is_terminal, {'forced_terminal': True, 'smiles': state.smiles}
        return 0.0, is_terminal, {'forced_terminal': False, 'smiles': state.smiles}
