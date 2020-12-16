import logging
import math
import random
from abc import abstractmethod
from typing import Iterable, Iterator, List, Optional

import numpy as np

from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.state import State

logger = logging.getLogger(__name__)


class MCTSNode(object):
    def __init__(self, state: State, game: Optional[MCTSGame] = None) -> None:
        """A node that coordinates 'vanilla' MCTS optimization. This class must be subclassed with a `compute_reward`
        function, and provided a State class that defines the allowable action space.

       For instance, to optimize QED of a molecule:
        >>> class QEDNode(MCTSNode):
        ...     def compute_reward(self):
        ...     return qed(self.state.molecule)

        :param state: The starting State instance that is used to initialize the tree search
        :param game: A MCTSGame instance that provides overall configuration parameters to all the nodes.
        """
        if game is None:
            game = MCTSGame()

        self._state = state
        self._game = game
        self._expanded: bool = False  # True iff node has been expanded

        self._visits: int = 0  # visit count
        self._total_value: float = 0.0
        self._reward: Optional[float] = None  # lazily initialized
        self._successors: Optional[List[MCTSNode]] = None

    @abstractmethod
    def compute_reward(self) -> float:
        pass

    @property
    def state(self) -> State:
        """
        :return: delegate which defines the graph structure being explored
        """
        return self._state

    @property
    def game(self) -> MCTSGame:
        """
        :return: delegate which contains game-level configuration
        """
        return self._game

    @property
    def expanded(self) -> bool:
        return self._expanded

    def expand(self) -> None:
        """self.get_successors will always return successor nodes, but these will not be explored in self.tree_policy
        unless self.expanded is True. """
        self._expanded = True

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def value(self) -> float:
        return self._total_value / self._visits if self._visits != 0 else 0

    def ucb_score(self, child: 'MCTSNode') -> float:
        game = self._game
        if self.visits == 0:
            raise RuntimeError("Child {} of parent {} with zero visits".format(child, self))
        if child.visits == 0:
            return math.inf
        return child.value + game.ucb_constant * math.sqrt(
            2 * math.log(self.visits) / child.visits)

    def update(self, reward: float) -> None:
        """
        Updates this node with a visit and a reward
        """
        self._visits += 1
        self._total_value += reward

    def __eq__(self, other: any) -> bool:
        """
        equals method delegates to self._graph_node for easy hashing based on graph structure
        """
        return isinstance(other, self.__class__) and self._state == other._state

    def __hash__(self) -> int:
        """
        hash method delegates to self._graph_node for easy hashing based on graph structure
        """
        return hash(self._state)

    def __repr__(self) -> str:
        """
        repr method delegates to self._graph_node
        """
        return self._state.__repr__()

    def get_successors(self) -> Iterable['MCTSNode']:
        """ Default implementation that caches successors locally. This routine can be overridden to provide more
        advanced memoization or graph resolution. """

        if self._successors is None:
            self._successors = [self.__class__(action, self.game)
                                for action in self._state.get_next_actions()]

        return self._successors

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    def tree_policy(self) -> Iterator['MCTSNode']:
        """
        Implements the tree search part of an MCTS search. Recursive function which
        returns a generator over the optimal path.
        """
        yield self
        if self.expanded:
            successor = max(self.get_successors(), key=lambda successor: self.ucb_score(successor))
            yield from successor.tree_policy()

    @property
    def reward(self) -> float:
        assert self.terminal, "Accessing reward of non-terminal state"
        if self._reward is None:
            self._reward = self.compute_reward()
        return self._reward

    def evaluate(self) -> float:
        """ In MCTS, we evaluate nodes through a random rollout of potential future actions.

        :return: reward of a terminal state selected from the current node
        """

        def random_rollout(state: State) -> float:
            """Recursively descend the action space until a final node is reached"""
            if state.terminal:
                return self.__class__(state).compute_reward()
            else:
                return random_rollout(random.choice(list(state.get_next_actions())))

        return random_rollout(self.state)

    def mcts_step(self) -> 'MCTSNode':
        """
        Perform a single MCTS step from the given starting node, including a
        tree search, expansion, and backpropagation.
        """

        # Perform the tree policy search
        history = list(self.tree_policy())
        leaf = history[-1]
        if (leaf.visits > 0) and not leaf.terminal:
            leaf.expand()

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
        successors = list(self.get_successors())
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
                choice = sorted((node for node in self.get_successors()),
                                key=lambda x: -x.visits)[0]

            yield from choice.run_mcts(num_simulations, explore=explore)
