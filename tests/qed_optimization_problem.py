import functools
from abc import ABC

import numpy as np
import rdkit
import sqlalchemy
from rdkit.Chem.QED import qed

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_problem import MoleculeProblem, MoleculeTFAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState


class QEDOptimizationProblem(MoleculeProblem, AlphaZeroProblem, ABC):

    @functools.lru_cache(maxsize=None)
    def get_reward(self, state: MoleculeState) -> (float, {}):
        if state.forced_terminal:
            return qed(state.molecule), {'forced_terminal': True}
        return 0.0, {'forced_terminal': False}


class QEDWithRandomPolicy(QEDOptimizationProblem):

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
        random_state = np.random.RandomState()
        children = parent.children
        priors = random_state.dirichlet(np.ones(len(children)))

        return random_state.random(), {vertex: prior for vertex, prior in zip(children, priors)}


class QEDWithMoleculePolicy(MoleculeTFAlphaZeroProblem, QEDOptimizationProblem):
    pass
