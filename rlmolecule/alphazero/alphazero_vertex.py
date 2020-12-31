import logging
from typing import (Dict, Optional)

from rlmolecule.mcts.mcts_vertex import MCTSVertex
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class AlphaZeroVertex(MCTSVertex):
    """
    A class which implements the AlphaZero search methodology, with the assistance of a supplied
    AlphaZeroGame implementation ("game").

    Users must implement a `policy` function, that takes as inputs the next possible actions and returns a value
    score for the current vertex and prior score for each child.
    """

    def __init__(self, state: GraphSearchState) -> None:
        super().__init__(state)

        self.child_priors: Optional[Dict['AlphaZeroVertex', float]] = None  # lazily initialized
        # self.policy_inputs: Optional[Dict[str, np.ndarray]] = None  # lazily initialized
        # self.policy_data = None  # lazily initialized
