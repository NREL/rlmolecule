import logging
from typing import Callable, Generator, Optional

import numpy as np

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts import MCTS
from rlmolecule.tree_search.graph_search import GraphSearch
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class AlphaZero(GraphSearch[AlphaZeroVertex]):
    """
    This class defines the interface for implementing AlphaZero-based games within this framework.
    Such a game overrides the abstract methods below with application-specific implementations.
    
    AlphaZeroGame interacts with AlphaZeroVertex. See AlphaZeroVertex for more details.
    """

    def __init__(self,
                 problem: AlphaZeroProblem,
                 min_reward: float = 0.0,
                 pb_c_base: float = 1.0,
                 pb_c_init: float = 1.25,
                 dirichlet_noise: bool = True,
                 dirichlet_alpha: float = 1.0,
                 dirichlet_x: float = 0.25,
                 num_mcts_samples: int = 20,
                 ) -> None:
        """
        Constructor.
        :param min_reward: Minimum reward to return for invalid actions
        :param pb_c_base: 19652 in pseudocode
        :param pb_c_init:
        :param dirichlet_noise: whether to add dirichlet noise
        :param dirichlet_alpha: dirichlet 'shape' parameter. Larger values spread out probability over more moves.
        :param dirichlet_x: percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise
        """
        super().__init__()
        self._problem = problem
        self._min_reward: float = min_reward
        self._pb_c_base: float = pb_c_base
        self._pb_c_init: float = pb_c_init
        self._dirichlet_noise: bool = dirichlet_noise
        self._dirichlet_alpha: float = dirichlet_alpha
        self._dirichlet_x: float = dirichlet_x
        self._num_mcts_samples: int = num_mcts_samples

    @property
    def problem(self) -> AlphaZeroProblem:
        return self._problem

    @property
    def min_reward(self) -> float:
        return self._min_reward

    @property
    def pb_c_base(self) -> float:
        return self._pb_c_base

    @property
    def pb_c_init(self) -> float:
        return self._pb_c_init

    @property
    def dirichlet_noise(self) -> bool:
        return self._dirichlet_noise

    @property
    def dirichlet_alpha(self) -> float:
        return self._dirichlet_alpha

    @property
    def dirichlet_x(self) -> float:
        return self._dirichlet_x

    def run(
            self,
            state: Optional[GraphSearchState] = None,
            explore: bool = True,
            num_mcts_samples: Optional[int] = None,
            mcts_selection_function: Optional[Callable[[AlphaZeroVertex], AlphaZeroVertex]] = None,
    ) -> Generator[AlphaZeroVertex]:
        vertex = self._get_root() if state is None else self.get_vertex_for_state(state)
        action_selection_function = MCTS.softmax_selection if explore else MCTS.visit_selection
        num_mcts_samples = self._num_mcts_samples if num_mcts_samples is None else num_mcts_samples

        def ucb_selection(parent):
            return max(parent.children, key=lambda child: self._ucb_score(parent, child))

        mcts_selection_function = ucb_selection if mcts_selection_function is None else mcts_selection_function

        # noinspection PyTypeChecker
        yield from self.run_from_vertex(vertex, action_selection_function, num_mcts_samples, mcts_selection_function)

    def run_from_vertex(
            self,
            vertex: AlphaZeroVertex,
            action_selection_function: Callable[[AlphaZeroVertex], AlphaZeroVertex],
            num_mcts_samples: int,
            mcts_selection_function: Callable[[AlphaZeroVertex], AlphaZeroVertex],
    ) -> Generator[AlphaZeroVertex]:
        yield vertex
        self.mcts_sample(vertex, num_mcts_samples, mcts_selection_function)
        # self.problem.store_policy_inputs_and_targets(state) # elided
        children = vertex.children
        if children is not None and len(children) > 0:
            next = action_selection_function(vertex)
            yield from self.run_from_vertex(next, action_selection_function, num_mcts_samples, mcts_selection_function)

    def mcts_sample(
            self,
            vertex: AlphaZeroVertex,
            num_mcts_samples: int,
            mcts_selection_function: Callable[[AlphaZeroVertex], AlphaZeroVertex],
    ) -> None:
        for _ in range(num_mcts_samples):
            search_path, value = self._select(vertex, mcts_selection_function)
            self._backpropagate(search_path, value)

    def _select(
            self,
            root: AlphaZeroVertex,
            mcts_selection_function: Callable[[AlphaZeroVertex], AlphaZeroVertex],
    ) -> ([AlphaZeroVertex], float):
        """
        Selection step of AlphaZero
        :return search path and value estimate
        """

        current = root
        search_path = []
        while True:
            search_path.append(current)
            children = current.children

            if children is None:  # node is unexpanded: expand and return its value estimate
                value = self._expand_and_evaluate(current)
                return search_path, value

            if len(children) == 0:  # node is terminal: return its value
                return search_path, self.problem.evaluate(current)[0]

            current = mcts_selection_function(current)

    def _expand_and_evaluate(self, parent: AlphaZeroVertex) -> float:
        """
        Expansion step of AlphaZero
        :return value estimate
        """

        # MCTS expansion
        children = [self.get_vertex_for_state(state) for state in parent.state.get_next_actions()]
        parent.children = children

        # get value estimate and child priors
        value, child_priors = self.problem.evaluate(parent)

        # Store prior values for child vertices predicted from the policy network, and add dirichlet noise as
        # specified in the game configuration.
        if self.dirichlet_noise != 0:
            prior_array: np.ndarray = np.array([child_priors[child] for child in children])
            random_state = np.random.RandomState()
            noise = random_state.dirichlet(np.ones_like(prior_array) * self.dirichlet_alpha)
            prior_array = prior_array * (1 - self.dirichlet_x) + (noise * self.dirichlet_x)
            child_priors = prior_array.tolist()
        normalization_factor = child_priors.sum()
        parent.child_priors = {child: prior / normalization_factor for child, prior in zip(children, child_priors)}

        return value

    @staticmethod
    def _backpropagate(search_path: [AlphaZeroVertex], value: float):
        """
        Backpropagation step of AlphaZero
        Update each vertex with value estimate along search path
        """
        for vertex in reversed(search_path):
            vertex.update(value)

    def _ucb_score(self, parent: AlphaZeroVertex, child: AlphaZeroVertex) -> float:
        """
        A modified upper confidence bound score for the vertices value, incorporating the prior prediction.

        :param child: Vertex for which the UCB score is desired
        :return: UCB score for the given child
        """
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * parent.child_priors[child]
        return prior_score + child.value

    def _get_root(self) -> AlphaZeroVertex:
        return self.get_vertex_for_state(self.problem.get_initial_state())
