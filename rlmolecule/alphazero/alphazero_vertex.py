import logging
from abc import abstractmethod
from typing import (
    List,
    Optional, Dict,
)

import numpy as np

from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class AlphaZeroVertex(MCTSVertex):
    """
    A class which implements the AlphaZero search methodology, with the assistance of a supplied
    AlphaZeroGame implementation ("game").

    Users must implement a `policy` function, that takes as inputs the next possible actions and returns a value
    score for the current vertex and prior score for each child.
    """

    def __init__(self, state: GraphSearchState) -> None:
        super().__init__(state)

        self.child_priors: Optional[Dict['AlphaZeroVertex', float]] = None  # lazily initialized
        self.policy_inputs: Optional[Dict[str, np.ndarray]] = None  # lazily initialized
        self.policy_data = None  # lazily initialized

    # @abstractmethod
    # def policy(self, children: List['MCTSVertex']) -> (float, Dict['MCTSVertex', float]):
    #     """
    #     A user-provided function to get value and prior estimates for the given vertex. Accepts a list of child
    #     vertices for the given state, and should return both the predicted value of the current vertex, as well as prior
    #     scores for each child vertex.
    #
    #     :param children: A list of AlphaZeroVertices corresponding to next potential actions
    #     :return: (value_of_current_vertex, {child_vertex: child_prior for child_vertex in children})
    #     """
    #     pass

    # def ucb_score(self, child: 'AlphaZeroVertex') -> float:
    #     """A modified upper confidence bound score for the vertices value, incorporating the prior prediction.
    #
    #     :param child: Vertex for which the UCB score is desired
    #     :return: UCB score for the given child
    #     """
    #     game: AlphaZeroGame = self._game
    #     pb_c = np.log((self.visits + game.pb_c_base + 1) / game.pb_c_base) + game.pb_c_init
    #     pb_c *= np.sqrt(self.visits) / (child.visits + 1)
    #     prior_score = pb_c * self.child_prior(child)
    #     return prior_score + child.value

    # def evaluate_and_expand(self) -> float:
    #     """In alphazero, these steps must happen simultaneously. For a given vertex, expand and run the policy network to
    #     get prior predictions and a value estimate.
    #
    #     Returns:
    #     :return: value (float), the estimated value of the current vertex
    #     """
    #     if self.terminal:
    #         return self.game.compute_reward(self)
    #
    #     MCTSVertex.expand(self)
    #
    #     # This will cause issues later, so we catch an incorrect state.terminal definition here
    #     assert self.expanded, f"{self} has no valid children, but is not a terminal state"
    #
    #     # noinspection PyTypeChecker
    #     value, child_priors = self.policy(self.children)
    #     self.store_child_priors_with_noise(child_priors)
    #
    #     return value

    # def store_child_priors_with_noise(self, child_priors: Dict['MCTSVertex', float]) -> None:
    #     """Store prior values for child vertices predicted from the policy network, and add dirichlet noise as
    #     specified in the game configuration.
    #
    #     :param child_priors: A dictionary of prior scores predicted by the policy network.
    #     """
    #     game: AlphaZeroGame = self._game
    #     children = self.children
    #     priors = np.array([child_priors[child] for child in children])
    #
    #     if game.dirichlet_noise:
    #         random_state = np.random.RandomState()
    #         noise = random_state.dirichlet(np.ones_like(priors) * game.dirichlet_alpha)
    #         priors = priors * (1 - game.dirichlet_x) + (noise * game.dirichlet_x)
    #
    #     assert np.isclose(priors.sum(), 1.), "child priors need to sum to one"
    #     self.child_priors = {child: prior for child, prior in zip(children, priors)}
