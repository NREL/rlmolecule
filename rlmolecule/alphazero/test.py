from typing import Dict, List

import numpy as np

from rlmolecule.alphazero.alphazero_node import AlphaZeroNode


class RandomPolicy(AlphaZeroNode):
    """A test policy that just returns random scores for both the value and prior predictions"""

    def policy(self, successors: List['MCTSNode']) -> (float, Dict['MCTSNode', float]):
        random_state = np.random.RandomState()
        priors = random_state.dirichlet(np.ones(len(successors)))

        return random_state.random(), {node: prior for node, prior in zip(successors, priors)}
