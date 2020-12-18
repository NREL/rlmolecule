import logging
from typing import Optional

# from rlmolecule.mcts.mcts_game import MCTSGame
from rlmolecule.tree_search.tree_search_node import TreeSearchNode
from rlmolecule.tree_search.tree_search_state import TreeSearchState

logger = logging.getLogger(__name__)


class MCTSNode(TreeSearchNode['MCTSNode']):

    def __init__(self, state: TreeSearchState) -> None:
        """A node that coordinates 'vanilla' MCTS optimization. This class must be subclassed with a `compute_reward`
        function, and provided a State class that defines the allowable action space.

        TODO: update documentation
       For instance, to optimize QED of a molecule:
        >>> class QEDNode(MCTSNode):
        ...     def compute_reward(self):
        ...     return qed(self.state.molecule)

        :param state: The starting State instance that is used to initialize the tree search
        :param game: A MCTSGame instance that provides overall configuration parameters to all the nodes.
        """
        super().__init__(state)
        self.reward: Optional[float] = None  # lazily initialized
