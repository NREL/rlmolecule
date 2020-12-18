import logging

from typing import (
    List,
    Optional,
)

import numpy as np

from rlmolecule.alphazero.alphazero_game import AlphaZeroGame
from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.tree_search.tree_search_state import TreeSearchState

logger = logging.getLogger(__name__)


class AlphaZeroNode(MCTSNode):
    """
    A class which implements the AlphaZero search methodology, with the assistance of a supplied
    AlphaZeroGame implementation ("game").

    Users must implement a `policy` function, that takes as inputs the next possible actions and returns a value
    score for the current node and prior score for each successor.
    """

    def __init__(self, state: TreeSearchState, game: AlphaZeroGame) -> None:
        super().__init__(state, game)

        self._successor_priors: Optional[Dict['MCTSNode', float]] = None  # lazily initialized
        self._policy_inputs: Optional[Dict[str, np.ndarray]] = None  # lazily initialized
        self._policy_data = None  # lazily initialized

    @abstractmethod
    def policy(self, successors: List['MCTSNode']) -> (float, Dict['MCTSNode', float]):
        """
        A user-provided function to get value and prior estimates for the given node. Accepts a list of successor
        nodes for the given state, and should return both the predicted value of the current node, as well as prior
        scores for each successor node.

        :param successors: A list of AlphaZeroNodes corresponding to next potential actions
        :return: (value_of_current_node, {successor_node: successor_prior for successor_node in successors})
        """
        pass

    def ucb_score(self, child: 'AlphaZeroNode') -> float:
        """A modified upper confidence bound score for the nodes value, incorporating the prior prediction.

        :param child: Node for which the UCB score is desired
        :return: UCB score for the given child
        """
        game: AlphaZeroGame = self._game
        pb_c = np.log((self.visits + game.pb_c_base + 1) / game.pb_c_base) + game.pb_c_init
        pb_c *= np.sqrt(self.visits) / (child.visits + 1)
        prior_score = pb_c * self.successor_prior(child)
        return prior_score + child.value

    def evaluate_and_expand(self) -> float:
        """In alphazero, these steps must happen simultaneously. For a given node, expand and run the policy network to
        get prior predictions and a value estimate.

        Returns:
        :return: value (float), the estimated value of the current node
        """
        if self.terminal:
            return self.reward

        MCTSNode.expand(self)

        # This will cause issues later, so we catch an incorrect state.terminal definition here
        assert self.expanded, f"{self} has no valid successors, but is not a terminal state"

        value, successor_priors = self.policy(self.successors)
        self.store_successor_priors_with_noise(successor_priors)

        return value

    def evaluate(self) -> float:
        return self.evaluate_and_expand()

    def expand(self) -> 'MCTSNode':
        self.evaluate_and_expand()
        return self

    def store_successor_priors_with_noise(self, successor_priors: Dict['MCTSNode', float]) -> None:
        """Store prior values for successor nodes predicted from the policy network, and add dirichlet noise as
        specified in the game configuration.

        :param successor_priors: A dictionary of prior scores predicted by the policy network.
        """
        game: AlphaZeroGame = self._game
        successors = self.successors
        priors = np.array([successor_priors[successor] for successor in successors])

        if game.dirichlet_noise:
            random_state = np.random.RandomState()
            noise = random_state.dirichlet(np.ones_like(priors) * game.dirichlet_alpha)
            priors = priors * (1 - game.dirichlet_x) + (noise * game.dirichlet_x)

        assert np.isclose(priors.sum(), 1.), "successor priors need to sum to one"
        self._successor_priors = {successor: prior for successor, prior in zip(successors, priors)}

    def successor_prior(self, child: 'AlphaZeroNode') -> float:
        """
        Prior probabilities (unlike logits) depend on the parent
        """
        return self._successor_priors[child]