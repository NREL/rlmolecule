from typing import Dict, List

import numpy as np

from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex


class RandomPolicy(AlphaZeroVertex):
    """A test policy that just returns random scores for both the value and prior predictions"""

    def policy(self, successors: List['MCTSVertex']) -> (float, Dict['MCTSVertex', float]):
        random_state = np.random.RandomState()
        priors = random_state.dirichlet(np.ones(len(successors)))

        return random_state.random(), {vertex: prior for vertex, prior in zip(successors, priors)}
