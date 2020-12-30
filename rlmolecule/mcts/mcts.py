import itertools
import logging
import math
import random
import sys
from typing import Callable, List, Optional, Type

import numpy as np

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
            num_mcts_samples: int = 1,
            action_selection_function: Optional[Callable[[MCTSVertex], MCTSVertex]] = None,
            max_depth: Optional[int] = None,
    ) -> ([], float):
        """Run the MCTS search from the given starting state (or the root node if not provided). This function runs a
        given number of MCTS iterations per step, and then recursively descends the action space according to the
        provided `action_selection_function` (softmax sampling of visit counts if not provided).

        todo: we might want to provide a max-depth option?

        :param num_mcts_samples: number of samples to perform at each level of the MCTS search
        :param state: the starting state, or if not provided, the state returned from _get_root()
        :param action_selection_function: a function used to select among the possible next actions. Defaults to
            softmax sampling by visit counts.
        :return: The search path (as a list of vertexes) and the final reward from the last state.
        :param max_depth: the maximum number of times to recurse. Defaults to system recursion limit.
        """
        vertex = self._get_root() if state is None else self.get_vertex_for_state(state)
        if action_selection_function is None:
            action_selection_function = self.softmax_selection

        path: [] = []
        value = 0.
        for depth in itertools.count():
            for _ in range(num_mcts_samples):
                value = self.sample_from(vertex)
            self._accumulate_path_data(vertex, path)
            children = vertex.children
            if children is None or len(children) == 0:
                break
            if depth > (max_depth if max_depth is not None else sys.getrecursionlimit()):
                logger.warning(f"{self} reached max_depth or recursion limit")
                break
            vertex = action_selection_function(vertex)

        return path, value

    def sample_from(self, vertex: MCTSVertex) -> float:
        """Run a single MCTS sample from the given vertex

        :param vertex: The starting vertex for MCTS sampling
        """
        search_path = self._select(vertex)
        value = self._evaluate(search_path[-1], search_path)
        self._backpropagate(search_path, value)
        return value

    # noinspection PyMethodMayBeStatic
    def _accumulate_path_data(self, vertex: MCTSVertex, path: []):
        path.append(vertex)

    def _select(self, root: MCTSVertex) -> [MCTSVertex]:
        """
        Selection step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Selection: Start from root R and select successive child vertices until a leaf vertex L is reached.
        The root is the current game state and a leaf is any vertex that has a potential child from which no simulation
        (playout) has yet been initiated. The section below says more about a way of biasing choice of child vertices that
        lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
        """

        def _iter_select(parent: MCTSVertex) -> [MCTSVertex]:
            yield parent
            if parent.expanded and len(parent.children) > 0:
                next_parent = max(parent.children, key=lambda child: self._ucb_score(parent, child))
                yield from _iter_select(next_parent)

        return list(_iter_select(root))

    def _expand(self, leaf: MCTSVertex) -> None:
        """
        Expansion step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child
        vertices and choose vertex C from one of them. Child vertices are any valid moves from the game position defined by L.
        """
        if not leaf.expanded:
            leaf.children = [self.get_vertex_for_state(state) for state in leaf.state.get_next_actions()]

    def _evaluate(self, leaf: MCTSVertex, search_path: [MCTSVertex],) -> float:
        """
        Estimates the value of a leaf vertex.
        Simulation step of MCTS.
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Simulation: Complete one random playout from vertex C. This step is sometimes also called playout or rollout.
        A playout may be as simple as choosing uniform random moves until the game is decided (for example in chess,
        the game is won, lost, or drawn).
        :return: value estimate of the given leaf vertex
        """

        # This `expand` call sets up further visits for this node, but visits to children
        # aren't tracked below the given leaf node
        self._expand(leaf)

        state = leaf.state
        children = state.get_next_actions()

        while len(children) > 0:
            state = random.choice(children)
            children = state.get_next_actions()

        return self.problem.get_reward(state)

    @staticmethod
    def _backpropagate(search_path: [MCTSVertex], value: float):
        """
        Backpropagation step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Backpropagation: Use the result of the playout to update information in the vertices on the search_path from C to R.
        """
        for vertex in reversed(search_path):
            vertex.update(value)

    @staticmethod
    def visit_selection(parent: MCTSVertex) -> MCTSVertex:
        return max(parent.children, key=lambda child: child.visit_count)

    @staticmethod
    def softmax_selection(parent: MCTSVertex) -> MCTSVertex:
        children: List[MCTSVertex] = parent.children
        visit_counts = np.array([child.visit_count for child in children])
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
        return child.value + self.ucb_constant * math.sqrt(2 * math.log(parent.visit_count) / child.visit_count)
