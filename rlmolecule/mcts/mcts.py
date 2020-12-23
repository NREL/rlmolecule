import math
import random
from typing import List, Callable

import numpy as np

from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search import GraphSearch
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTS(GraphSearch[MCTSVertex]):
    def __init__(
            self,
            problem: MCTSProblem,
            ucb_constant: float = math.sqrt(2),
    ) -> None:
        super().__init__()
        self._problem: MCTSProblem = problem
        self.ucb_constant = ucb_constant

    @property
    def problem(self) -> MCTSProblem:
        return self._problem

    def run(self, iterations: int, explore: bool = True) -> None:
        selection_function = self.visit_selection
        if explore:
            def ucb_selection(parent):
                return max(parent.children, key=lambda child: self._ucb_score(parent, child))
            selection_function = ucb_selection

        for _ in range(iterations):
            self.step(selection_function)

    def step(self, selection_function: Callable[[MCTSVertex], MCTSVertex]) -> None:
        root: MCTSVertex = self._get_root()
        search_path, terminal_state = self._select(root, selection_function)
        value = self.problem.evaluate(terminal_state)
        self._backpropagate(search_path, value)

    def _select(
            self,
            root: MCTSVertex,
            selection_function: Callable[[MCTSVertex], MCTSVertex],
    ) -> [MCTSVertex]:
        """
        Selection step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Selection: Start from root R and select successive child vertices until a leaf vertex L is reached.
        The root is the current game state and a leaf is any vertex that has a potential child from which no simulation
        (playout) has yet been initiated. The section below says more about a way of biasing choice of child vertices that
        lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
        """
        current = root
        search_path = []
        while True:
            search_path.append(current)
            children = current.children

            if children is None:
                # node is unexpanded: expand it, choose a child, and simulate from that child to a terminal state
                children = self._expand(current)
                if len(children) > 0:
                    child = selection_function(current)
                    search_path.append(child)
                    return search_path, self._simulate(child.state)

            if len(children) == 0:
                # node is expanded and terminal
                return search_path, current.state

            current = selection_function(current)

    def _expand(self, leaf: MCTSVertex) -> [MCTSVertex]:
        """
        Expansion step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child
        vertices and choose vertex C from one of them. Child vertices are any valid moves from the game position defined by L.
        """
        children = [self.get_vertex_for_state(state) for state in leaf.state.get_next_actions()]
        leaf.children = children
        return children

    @staticmethod
    def _simulate(start: GraphSearchState) -> GraphSearchState:
        """
        Simulation step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Simulation: Complete one random playout from vertex C. This step is sometimes also called playout or rollout.
        A playout may be as simple as choosing uniform random moves until the game is decided (for example in chess,
        the game is won, lost, or drawn).
        """
        current = start
        while True:
            children = current.get_next_actions()
            if len(children) == 0:
                break
            current = random.choice(children)
        return current

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
