import math
import random
from typing import List, Callable, Generator, Optional

import numpy as np

from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search import GraphSearch
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class MCTS(GraphSearch[MCTSVertex]):
    def __init__(
            self,
            problem: MCTSProblem,
            num_mcts_samples: int = 20,
            ucb_constant: float = math.sqrt(2),
    ) -> None:
        super().__init__()
        self._problem: MCTSProblem = problem
        self._num_mcts_samples: int = num_mcts_samples
        self.ucb_constant: float = ucb_constant

    @property
    def problem(self) -> MCTSProblem:
        return self._problem

    def run(
            self,
            state: Optional[GraphSearchState] = None,
            explore: bool = True,
            num_mcts_samples: Optional[int] = None,
            mcts_selection_function: Optional[Callable[[MCTSVertex], MCTSVertex]] = None,
    ) -> Generator[MCTSVertex]:
        vertex = self._get_root() if state is None else self.get_vertex_for_state(state)
        action_selection_function = self.softmax_selection if explore else self.visit_selection
        num_mcts_samples = self._num_mcts_samples if num_mcts_samples is None else num_mcts_samples

        def ucb_selection(parent):
            return max(parent.children, key=lambda child: self._ucb_score(parent, child))

        mcts_selection_function = ucb_selection if mcts_selection_function is None else mcts_selection_function

        # noinspection PyTypeChecker
        yield from self.run_from_vertex(vertex, action_selection_function, num_mcts_samples, mcts_selection_function)

    def run_from_vertex(
            self,
            vertex: MCTSVertex,
            action_selection_function: Callable[[MCTSVertex], MCTSVertex],
            num_mcts_samples: int,
            mcts_selection_function: Callable[[MCTSVertex], MCTSVertex],
    ) -> Generator[MCTSVertex]:
        yield vertex
        self.sample(vertex, num_mcts_samples, mcts_selection_function)
        # self.problem.store_policy_inputs_and_targets(state) # elided
        children = vertex.children
        if children is not None and len(children) > 0:
            child = action_selection_function(vertex)
            yield from self.run_from_vertex(child, action_selection_function, num_mcts_samples, mcts_selection_function)

    def sample(
            self,
            vertex: MCTSVertex,
            num_mcts_samples: int,
            mcts_selection_function: Callable[[MCTSVertex], MCTSVertex],
    ) -> None:
        for _ in range(num_mcts_samples):
            search_path, value = self._select(vertex, mcts_selection_function)
            self._backpropagate(search_path, value)

    def _select(
            self,
            root: MCTSVertex,
            mcts_selection_function: Callable[[MCTSVertex], MCTSVertex],
    ) -> ([MCTSVertex], float):
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

            if children is None:  # node is unexpanded: expand and return its value estimate
                return search_path, self._expand_and_evaluate(current, mcts_selection_function, search_path)

            if len(children) == 0:  # node is expanded and terminal: return its value
                return search_path, self.problem.get_reward(current.state)

            current = mcts_selection_function(current)

    def _expand_and_evaluate(
            self,
            leaf: MCTSVertex,
            search_path: [MCTSVertex],
    ) -> float:
        """
        Expansion step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child
        vertices and choose vertex C from one of them. Child vertices are any valid moves from the game position defined by L.
        """
        children = [self.get_vertex_for_state(state) for state in leaf.state.get_next_actions()]
        leaf.children = children

        state = leaf.state
        if len(children) > 0:
            child = random.choice(children)
            search_path.append(child)
            state = self._simulate(child.state)

        return self.problem.get_reward(state)

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
