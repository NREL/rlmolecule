import logging
import math
import random
from typing import Iterator, List, Optional

import numpy as np

# from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.tree_search.tree_search_game import TreeSearchGame
from rlmolecule.tree_search.tree_search_node import TreeSearchNode
from rlmolecule.tree_search.tree_search_state import TreeSearchState

logger = logging.getLogger(__name__)


class MCTSNode(TreeSearchNode):
    def __init__(self, state: TreeSearchState, game: 'MCTSGame') -> None:
        """A node that coordinates 'vanilla' MCTS optimization. This class must be subclassed with a `compute_reward`
        function, and provided a State class that defines the allowable action space.

        TODO: update documentation
       For instance, to optimize QED of a molecule:
        >>> class QEDNode(MCTSNode):
        ...     def compute_reward(self):
        ...     return qed(self.state.molecule)

        :param state: The starting State instance that is used to initialize the tree search
        :param game: A MCTSGame instance that provides overall configuration parameters to all the nodes.
        """
        super().__init__(state, game)

        self._reward: Optional[float] = None  # lazily initialized

    def ucb_score(self, child: 'MCTSNode') -> float:
        """Calculates the UCB1 score for the given child node. From Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
        Machine Learning, 47(2/3), 235–256. doi:10.1023/a:1013689704352

        :param child: Node for which the UCB score is desired
        :return: UCB1 score.
        """

        # noinspection PyTypeChecker
        game: 'MCTSGame' = self._game
        if self.visits == 0:
            raise RuntimeError("Child {} of parent {} with zero visits".format(child, self))
        if child.visits == 0:
            return math.inf
        return child.value + game.ucb_constant * math.sqrt(
            2 * math.log(self.visits) / child.visits)

    def update(self, reward: float) -> 'MCTSNode':
        """
        Updates this node with a visit and a reward
        """
        self._visits += 1
        self._total_value += reward
        return self

    def tree_policy(self) -> Iterator['MCTSNode']:
        """
        Implements the tree search part of an MCTS search. Recursive function which
        returns a generator over the optimal path.
        """
        yield self
        if self.expanded:
            successor = max(self.successors, key=lambda successor: self.ucb_score(successor))

            # noinspection PyUnresolvedReferences
            yield from successor.tree_policy()

    @property
    def reward(self) -> float:
        if self._reward is None:
            self._reward = self._game.compute_reward(self)
        return self._reward

    def evaluate(self) -> float:
        """In MCTS, we evaluate nodes through a random rollout of potential future actions.

        :return: reward of a terminal state selected from the current node
        """

        # In MCTS, we only expand if the node has previously been visited
        if (self.visits > 0) and not self.terminal:
            self.expand()

        def random_rollout(node: MCTSNode) -> float:
            """Recursively descend the action space until a final node is reached"""
            if node.terminal:
                return node.reward
            else:
                # noinspection PyTypeChecker
                return random_rollout(random.choice(node.successors))

        return random_rollout(self)

    def mcts_step(self) -> 'MCTSNode':
        """
        Perform a single MCTS step from the given starting node, including a
        tree search, expansion, and backpropagation.
        """

        # Perform the tree policy search
        history = list(self.tree_policy())
        leaf = history[-1]

        value = leaf.evaluate()

        # perform backprop
        for node in history:
            node.update(value)

        return leaf

    def softmax_sample(self) -> 'MCTSNode':
        """
        Sample from successors according to their visit counts.
        Returns:
            choice: Node, the chosen successor node.
        """
        # noinspection PyTypeChecker
        successors: List[MCTSNode] = self.successors
        visit_counts = np.array([n._visits for n in successors])
        visit_softmax = np.exp(visit_counts) / sum(np.exp(visit_counts))
        return successors[np.random.choice(range(len(successors)), size=1, p=visit_softmax)[0]]

    def run_mcts(self, num_simulations: int, explore: bool = True) -> Iterator['MCTSNode']:
        """
        Performs a full game simulation, running num_simulations per iteration,
        choosing nodes either deterministically (explore=False) or via softmax sampling
        (explore=True) for subsequent iterations.
        Called recursively, returning a generator of game positions:
        >>> game = list(start.run_mcts(explore=True))

        :param num_simulations: Number of simulations to perform per MCTS step
        :param explore: whether to use softmax sampling (on visit counts) in choosing the next node, or to simply
            choose the node with the highest number of visits.
        """

        logger.info(
            f"{self._game.id}: selecting node {self.state} with value={self.value:.3f} and visits={self.visits}")

        yield self

        if not self.terminal:
            for _ in range(num_simulations):
                self.mcts_step()

            if explore:
                choice = self.softmax_sample()

            else:
                choice = sorted((node for node in self.successors), key=lambda x: -x.visits)[0]

            yield from choice.run_mcts(num_simulations, explore=explore)

    def _make_successor(self, action: TreeSearchState) -> 'MCTSNode':
        # noinspection PyTypeChecker
        return MCTSNode(action, self._game)
