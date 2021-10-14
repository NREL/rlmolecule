from collections import Sequence

from examples.gym.gridworld_env import GridWorldEnv
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class GridWorldGraphState(GraphSearchState):

    def __init__(self, delegate: GridWorldEnv):
        self.delegate: GridWorldEnv = delegate

    def equals(self, other: 'GraphSearchState') -> bool:
        return self.delegate == other.delegate

    def hash(self) -> int:
        return self.delegate.hash()

    def get_next_actions(self) -> Sequence['GridWorldGraphState']:
        valid_actions = []
        for action in range(len(self.delegate.action_map)):
            next, valid = self.delegate.make_next(action)
            if valid:
                valid_actions.append(next)
        return valid_actions


