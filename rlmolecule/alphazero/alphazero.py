import logging
import math

import numpy as np

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts import MCTS
from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.reward import Reward

logger = logging.getLogger(__name__)


class AlphaZero(MCTS):
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
                 ) -> None:
        """
        Constructor.
        :param min_reward: Minimum reward to return for invalid actions
        :param pb_c_base:
        :param pb_c_init:
        :param dirichlet_noise: whether to add dirichlet noise
        :param dirichlet_alpha: dirichlet 'shape' parameter. Larger values spread out probability over more moves.
        :param dirichlet_x: percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise
        """
        super().__init__(problem, vertex_class=AlphaZeroVertex)
        self._min_reward: float = min_reward
        self._pb_c_base: float = pb_c_base
        self._pb_c_init: float = pb_c_init
        self._dirichlet_noise: bool = dirichlet_noise
        self._dirichlet_alpha: float = dirichlet_alpha
        self._dirichlet_x: float = dirichlet_x

    @property
    def problem(self) -> AlphaZeroProblem:
        # noinspection PyTypeChecker
        return self._problem

    def _accumulate_path_data(self, vertex: MCTSVertex, path: []):
        children = vertex.children
        visit_sum = sum(child.visit_count for child in children)
        child_visits = [
            child.visit_count / visit_sum
            for child in children
        ]
        path.append((vertex, child_visits))

    def _evaluate(
            self,
            search_path: [AlphaZeroVertex],
    ) -> Reward:
        """
        Expansion step of AlphaZero, overrides MCTS evaluate step.
        Estimates the value of a leaf vertex.
        """
        assert len(search_path) > 0, 'Invalid attempt to evaluate an empty search path.'
        leaf = search_path[-1]
        self._expand(leaf)

        children = leaf.children
        if len(children) == 0:
            return self.problem.reward_wrapper(leaf.state)

        # get value estimate and child priors
        value, child_priors = self.problem.get_value_and_policy(leaf)

        # Store prior values for child vertices predicted from the policy network, and add dirichlet noise as
        # specified in the game configuration.
        prior_array: np.ndarray = np.array([child_priors[child] for child in children])

        if self._dirichlet_noise:
            random_state = np.random.RandomState()
            noise = random_state.dirichlet(np.ones_like(prior_array) * self._dirichlet_alpha)
            prior_array = prior_array * (1 - self._dirichlet_x) + (noise * self._dirichlet_x)

        child_priors = prior_array.tolist()
        normalization_factor = sum(child_priors)
        leaf.child_priors = {child: prior / normalization_factor for child, prior in zip(children, child_priors)}

        return self.problem.reward_class(value)

    def run(self, *args, **kwargs) -> ([], float):
        path, reward = MCTS.run(self, *args, **kwargs)
        self.problem._store_search_statistics(path, reward)
        return path, reward

    def _ucb_score(self, parent: AlphaZeroVertex, child: AlphaZeroVertex) -> float:
        """
        A modified upper confidence bound score for the vertices value, incorporating the prior prediction.

        :param child: Vertex for which the UCB score is desired
        :return: UCB score for the given child
        """
        child_priors = parent.child_priors
        if child_priors is None:
            return math.inf

        pb_c = np.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child_priors[child]
        return prior_score + child.value

    def _make_new_vertex(self, state: GraphSearchState) -> AlphaZeroVertex:
        return AlphaZeroVertex(state)
