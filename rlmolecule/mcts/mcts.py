import logging
import math
import random
from typing import Callable, List, Optional, Type

import numpy as np

from rlmolecule.tree_search.reward import Reward
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search import GraphSearch
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class MCTS(GraphSearch[MCTSVertex]):
    def __init__(
            self,
            problem: MCTSProblem,
            ucb_constant: float = math.sqrt(2),
            vertex_class: Optional[Type[MCTSVertex]] = None,
    ) -> None:
        super().__init__(MCTSVertex if vertex_class is None else vertex_class)
        self._problem: MCTSProblem = problem
        self.ucb_constant: float = ucb_constant

    @property
    def problem(self) -> MCTSProblem:
        return self._problem

    def run(
            self,
            state: Optional[GraphSearchState] = None,
            num_mcts_samples: int = 256,
            max_depth: int = 1000000,
            action_selection_function: Optional[Callable[[MCTSVertex], MCTSVertex]] = None,
            reset_canonicalizer: bool = True,
    ) -> ([], float):
        """
        Run the MCTS search from the given starting state (or the root node if not provided). This function runs a
        given number of MCTS iterations per step, and then descends the action space according to the
        provided `action_selection_function` (softmax sampling of visit counts if not provided).

        :param num_mcts_samples: number of samples to perform at each level of the MCTS search
        :param max_depth: the maximum search depth.
        :param state: the starting state, or if not provided, the state returned from _get_root()
        :param action_selection_function: a function used to select among the possible next actions. Defaults to
            softmax sampling by visit counts.
        :param reset_canonicalizer: whether to reset the graph canonicalizer in advance of the run
        :return: The search path (as a list of vertexes) and the reward from this search.
        """
        self.problem.initialize_run()

        if reset_canonicalizer:
            self.canonicalizer.reset()

        vertex = self._get_root() if state is None else self.get_vertex_for_state(state)
        action_selection_function = action_selection_function if action_selection_function is not None \
            else self.softmax_selection

        path: [] = []
        for _ in range(max_depth):
            # todo: this loop is odd, we're sampling terminal nodes a whole bunch of extra times
            self.sample(vertex, num_mcts_samples)
            self._accumulate_path_data(vertex, path)
            if len(vertex.children) == 0:
                return path, self.problem.reward_wrapper(vertex.state)
            logger.debug(f'{vertex} has children { {child: (round(child.value, 2), child.visit_count) for child in vertex.children} }')
            vertex = action_selection_function(vertex)

        logger.warning(f"{self} reached max_depth.")
        return path, math.nan  # todo: make sure this returns a reward class

    def sample(
            self,
            vertex: MCTSVertex,
            num_mcts_samples: int = 1,
    ) -> None:
        """
        Perform MCTS sampling from the given vertex.
        """
        for _ in range(num_mcts_samples):
            search_path = self._select(vertex)
            value = self._evaluate(search_path)
            self._backpropagate(search_path, value)

    # noinspection PyMethodMayBeStatic
    def _accumulate_path_data(self, vertex: MCTSVertex, path: []):
        path.append(vertex)

    def _select(
            self,
            root: MCTSVertex,
    ) -> [MCTSVertex]:
        """
        Selection step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Selection: Start from root R and select successive child vertices until a leaf vertex L is reached.
        The root is the current game state and a leaf is any vertex that has a potential child from which no simulation
        (playout) has yet been initiated. The section below says more about a way of biasing choice of child vertices that
        lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
        """
        search_path = [root]
        while True:
            current = search_path[-1]
            children = current.children
            if children is None or len(children) == 0:
                return search_path
            search_path.append(max(children, key=lambda child: self._ucb_score(current, child)))

    def _expand(self, leaf: MCTSVertex) -> None:
        """
        Expansion step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child
        vertices and choose vertex C from one of them. Child vertices are any valid moves from the game position defined by L.
        """
        if leaf.children is None:
            leaf.children = [self.get_vertex_for_state(state) for state in leaf.state.get_next_actions()]

    def _evaluate(
            self,
            search_path: [MCTSVertex],
    ) -> Reward:
        """
        Estimates the value of a leaf vertex.
        Simulation step of MCTS.
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Simulation: Complete one random playout from vertex C. This step is sometimes also called playout or rollout.
        A playout may be as simple as choosing uniform random moves until the game is decided (for example in chess,
        the game is won, lost, or drawn).
        :return: value estimate of the given leaf vertex
        """
        assert len(search_path) > 0, 'Invalid attempt to evaluate an empty search path.'
        leaf = search_path[-1]

        # This `expand` call sets up further visits for this node, but visits to children
        # aren't tracked below the given leaf node
        self._expand(leaf)

        state = leaf.state
        while True:
            children = state.get_next_actions()
            if len(children) == 0:
                return self.problem.reward_wrapper(state)
            state = random.choice(children)

    @staticmethod
    def _backpropagate(search_path: [MCTSVertex], value: Reward):
        """
        Backpropagation step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Backpropagation: Use the result of the playout to update information in the vertices on the search_path from C to R.
        """
        for vertex in reversed(search_path):
            vertex.update(value.scaled_reward)

    @staticmethod
    def visit_selection(parent: MCTSVertex) -> MCTSVertex:
        return max(parent.children, key=lambda child: child.visit_count)

    @staticmethod
    def softmax_selection(parent: MCTSVertex) -> MCTSVertex:
        children: List[MCTSVertex] = parent.children
        visit_counts = np.array([child.visit_count for child in children])
        visit_counts -= visit_counts.max()
        visit_softmax = np.exp(visit_counts) / sum(np.exp(visit_counts))
        return children[np.random.choice(range(len(children)), size=1, p=visit_softmax)[0]]

    def _get_root(self) -> MCTSVertex:
        return self.get_vertex_for_state(self.problem.get_initial_state())

    def _ucb_score(self, parent: MCTSVertex, child: MCTSVertex) -> float:
        """Calculates the UCB1 score for the given child vertex. From Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
           Machine Learning, 47(2/3), 235–256. doi:10.1023/a:1013689704352

           :param child: Vertex for which the UCB score is desired
           :return: UCB1 score.
           """
        if parent.visit_count == 0:
            raise RuntimeError("Child {} of parent {} with zero visits".format(child, self))
        if child.visit_count == 0:
            return math.inf
        return child.value + self.ucb_constant * math.sqrt(math.log(parent.visit_count) / child.visit_count)
