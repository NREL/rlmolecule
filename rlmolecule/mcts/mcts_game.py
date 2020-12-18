import math
import random
from typing import List, Callable

import numpy as np

from rlmolecule.mcts.mcts_node import MCTSNode
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.tree_search.tree_search_game import TreeSearchGame
from rlmolecule.tree_search.tree_search_state import TreeSearchState


class MCTSGame(TreeSearchGame[MCTSNode]):
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

    def run(self, explore: bool = True):
        root: MCTSNode = self._make_root()
        selection_function = self.visit_selection
        if explore:
            selection_function = self.ucb_selection

        path, terminal_state = self._select(root, selection_function)
        value = self.problem.compute_reward(terminal_state)
        self._backpropagate(path, value)

    def _select(self, root: MCTSNode, selection_function: Callable[[MCTSNode], MCTSNode]) -> [MCTSNode]:
        """
        Selection step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Selection: Start from root R and select successive child nodes until a leaf node L is reached.
        The root is the current game state and a leaf is any node that has a potential child from which no simulation
        (playout) has yet been initiated. The section below says more about a way of biasing choice of child nodes that
        lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
        """
        current = root
        path = []
        while True:
            path.append(current)
            children = current.children
            if children is None:
                children = self._expand(current)
                if len(children) > 0:
                    child = random.choice(children)
                    path.append(child)
                    return path, self._simulate(child.state)

            if len(children) == 0:
                return path, current.state
            current = selection_function(current)

    def _expand(self, leaf: MCTSNode) -> [MCTSNode]:
        """
        Expansion step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child
        nodes and choose node C from one of them. Child nodes are any valid moves from the game position defined by L.
        """
        children = [self._get_node_for_state(state) for state in leaf.state.get_next_actions()]
        leaf.children = children
        return children

    @staticmethod
    def _simulate(start: TreeSearchState) -> TreeSearchState:
        """
        Simulation step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Simulation: Complete one random playout from node C. This step is sometimes also called playout or rollout.
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
    def _backpropagate(path: [MCTSNode], value: float):
        """
        Backpropagation step of MCTS
        From Wikipedia (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search):
        Backpropagation: Use the result of the playout to update information in the nodes on the path from C to R.
        """
        for node in reversed(path):
            node.update(value)

    @staticmethod
    def ucb_selection(node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: child.visits)

    @staticmethod
    def visit_selection(node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: child.visits)

    @staticmethod
    def softmax_selection(node: MCTSNode) -> MCTSNode:
        children: List[MCTSNode] = node.children
        visit_counts = np.array([n.visits for n in children])
        visit_softmax = np.exp(visit_counts) / sum(np.exp(visit_counts))
        return children[np.random.choice(range(len(children)), size=1, p=visit_softmax)[0]]

    def _make_root(self) -> MCTSNode:
        return self._get_node_for_state(self.problem.get_initial_state())

    def _make_new_node(self, state: TreeSearchState) -> MCTSNode:
        return MCTSNode(state)

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculates the UCB1 score for the given child node. From Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
           Machine Learning, 47(2/3), 235–256. doi:10.1023/a:1013689704352

           :param child: Node for which the UCB score is desired
           :return: UCB1 score.
           """
        if parent.visits == 0:
            raise RuntimeError("Child {} of parent {} with zero visits".format(child, self))
        if child.visits == 0:
            return math.inf
        return child.value + self.ucb_constant * math.sqrt(2 * math.log(parent.visits) / child.visits)
