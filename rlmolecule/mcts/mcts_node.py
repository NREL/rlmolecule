import math
from abc import abstractmethod
import random
from typing import Iterable, Iterator, List, Optional

from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.state import State

class MCTSNode:
    def __init__(self, state: State, game: Optional[MCTSGame] = None) -> None:

        if game is None:
            game = MCTSGame()

        self._state = state
        self._game = game
        self.expanded: bool = False  # True iff node has been expanded

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
        return isinstance(other, 'MCTSNode') and self._state == other._state

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

    def evaluate(self):
        if self.terminal or (random.random() < self.game.early_stop_frac):
            return self.reward
        else:
            return random.choice(self.get_successors()).evaluate()

    def mcts_step(self) -> 'MCTSNode':
        """
        Perform a single MCTS step from the given starting node, including a
        tree search, expansion, and backpropagation.
        """

        # Perform the tree policy search
        history = list(self.tree_policy())
        leaf = history[-1]
        if (leaf.visits > 0) and not leaf.terminal:
            leaf.expanded = True

        value = leaf.evaluate()

        # perform backprop
        for node in history:
            node.update(value)

        return leaf