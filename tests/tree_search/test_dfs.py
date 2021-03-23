from typing import Sequence

import pytest

from rlmolecule.mcts.mcts import MCTS
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.tree_search.dfs import GraphCycleError
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory


class CyclicState(GraphSearchState):
    def __init__(self, position, is_terminal):
        super(CyclicState, self).__init__()
        self.is_terminal = is_terminal
        self.position = position

    def equals(self, other: 'CyclicState') -> bool:
        return (self.position == other.position) & (self.is_terminal == other.is_terminal)

    def __repr__(self):
        return f"{self.__class__}(position={self.position}, is_terminal={self.is_terminal})"

    def get_next_actions(self) -> Sequence['GraphSearchState']:
        if self.is_terminal:
            return []

        else:
            return [CyclicState(self.position, is_terminal=True),
                    CyclicState((self.position + 1) % 5, is_terminal=False)]

    def hash(self) -> int:
        return hash((self.position, self.is_terminal))


class AcyclicState(CyclicState):
    def get_next_actions(self) -> Sequence['GraphSearchState']:
        if self.is_terminal:
            return []

        else:
            return [AcyclicState(self.position, is_terminal=True),
                    AcyclicState(self.position + 1, is_terminal=False)]


class CyclicProblem(MCTSProblem):

    def get_initial_state(self) -> GraphSearchState:
        return CyclicState(0, is_terminal=False)

    def get_reward(self, state: CyclicState) -> (float, {}):
        return state.position, {}


class AcyclicProblem(CyclicProblem):

    def get_initial_state(self) -> GraphSearchState:
        return AcyclicState(0, is_terminal=False)


def test_dfs():

    game = MCTS(CyclicProblem(reward_class=LinearBoundedRewardFactory(min_reward=0, max_reward=5)))
    with pytest.raises(GraphCycleError):
        game.run(num_mcts_samples=100)

    # This shouldn't raise an error
    game = MCTS(AcyclicProblem(reward_class=LinearBoundedRewardFactory(min_reward=0, max_reward=5)))
    game.run(num_mcts_samples=100)
