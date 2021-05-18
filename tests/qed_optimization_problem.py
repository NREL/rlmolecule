import functools
from abc import ABC

import numpy as np
from rdkit.Chem.QED import qed

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.molecule_problem import MoleculeProblem, MoleculeTFAlphaZeroProblem
from rlmolecule.molecule.molecule_state import MoleculeState
from rlmolecule.tree_search.metrics import collect_metrics


class QEDOptimizationProblem(MoleculeProblem, AlphaZeroProblem, ABC):
    @collect_metrics
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

    def get_policy_inputs(self, state: 'GraphSearchState') -> {str: np.ndarray}:
        # these will get stored and used to index reward values
        return {'test': np.random.randn(5, 3)}


class QEDWithMoleculePolicy(MoleculeTFAlphaZeroProblem, QEDOptimizationProblem):
    pass
