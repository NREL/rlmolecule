import logging

import numpy as np

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.mcts.mcts import MCTS
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class AlphaZero(MCTS):
    """
    This class defines the interface for implementing AlphaZero-based games within this framework.
    Such a game overrides the abstract methods below with application-specific implementations.
    
    AlphaZeroGame interacts with AlphaZeroVertex. See AlphaZeroVertex for more details.
    """

    def __init__(self,
                 problem: AlphaZeroProblem,
                 num_mcts_samples: int = 20,
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
        :param pb_c_base: 19652 in pseudocode
        :param pb_c_init:
        :param dirichlet_noise: whether to add dirichlet noise
        :param dirichlet_alpha: dirichlet 'shape' parameter. Larger values spread out probability over more moves.
        :param dirichlet_x: percentage to favor dirichlet noise vs. prior estimation. Smaller means less noise
        """
        super().__init__(problem, num_mcts_samples=num_mcts_samples)
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

    def _evaluate(
            self,
            leaf: AlphaZeroVertex,
            search_path: [AlphaZeroVertex],
    ) -> float:
        """
        Expansion step of AlphaZero, overrides MCTS expansion step
        :return value estimate
        """
        children = leaf.children
        if len(children) == 0:
            return self.problem.get_reward(leaf.state)

        # get value estimate and child priors
        value, child_priors = self.problem.get_value_estimate(leaf)

        # Store prior values for child vertices predicted from the policy network, and add dirichlet noise as
        # specified in the game configuration.
        if self._dirichlet_noise != 0:
            prior_array: np.ndarray = np.array([child_priors[child] for child in children])
            random_state = np.random.RandomState()
            noise = random_state.dirichlet(np.ones_like(prior_array) * self._dirichlet_alpha)
            prior_array = prior_array * (1 - self._dirichlet_x) + (noise * self._dirichlet_x)
            child_priors = prior_array.tolist()
        normalization_factor = child_priors.sum()
        leaf.child_priors = {child: prior / normalization_factor for child, prior in zip(children, child_priors)}

        return value

    def _ucb_score(self, parent: AlphaZeroVertex, child: AlphaZeroVertex) -> float:
        """
        A modified upper confidence bound score for the vertices value, incorporating the prior prediction.

        :param child: Vertex for which the UCB score is desired
        :return: UCB score for the given child
        """
        pb_c = np.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * parent.child_priors[child]
        return prior_score + child.value

    def _make_new_vertex(self, state: GraphSearchState) -> AlphaZeroVertex:
        return AlphaZeroVertex(state)
