import functools

import numpy as np
import rdkit
import sqlalchemy
from rdkit.Chem.QED import qed

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.molecule.molecule_problem import MoleculeAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState


class QEDOptimizationProblem(AlphaZeroProblem):

    def __init__(self, engine: sqlalchemy.engine.Engine, config: MoleculeConfig, **kwargs) -> None:
        super(QEDOptimizationProblem, self).__init__(engine, **kwargs)
        self._config = config

    def get_initial_state(self) -> MoleculeState:
        return MoleculeState(rdkit.Chem.MolFromSmiles('C'), self._config)

    @functools.lru_cache(maxsize=None)
    def get_reward(self, state: MoleculeState) -> (float, {}):
        if state.forced_terminal:
            return qed(state.molecule), {'forced_terminal': True}
        return 0.0, {'forced_terminal': False}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        random_state = np.random.RandomState()
        children = parent.children
        priors = random_state.dirichlet(np.ones(len(children)))

        return random_state.random(), {vertex: prior for vertex, prior in zip(children, priors)}


class QEDWithMoleculePolicy(QEDOptimizationProblem, MoleculeAlphaZeroProblem):
    pass