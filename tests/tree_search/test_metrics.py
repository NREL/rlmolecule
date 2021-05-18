import time
from typing import Sequence

from rlmolecule.mcts.mcts import MCTS
from rlmolecule.mcts.mcts_problem import MCTSProblem
from rlmolecule.tree_search.graph_search_state import GraphSearchState
from rlmolecule.tree_search.metrics import call_metrics, collect_metrics
from rlmolecule.tree_search.reward import RawRewardFactory


class DummyState(GraphSearchState):
    def __init__(self, position, is_terminal):
        super(DummyState, self).__init__()
        self.is_terminal = is_terminal
        self.position = position

    def equals(self, other: 'CyclicState') -> bool:
        return (self.position == other.position) & (self.is_terminal == other.is_terminal)

    def __repr__(self):
        return f"{self.__class__}(position={self.position}, is_terminal={self.is_terminal})"

    @collect_metrics
    def get_next_actions(self) -> Sequence['GraphSearchState']:

        time.sleep(0.1)

        if self.is_terminal:
            return []

        else:
            return [
                DummyState(self.position, is_terminal=True),
                DummyState((self.position + 1), is_terminal=False)
            ]

    def hash(self) -> int:
        return hash((self.position, self.is_terminal))


class DummyProblem(MCTSProblem):
    def __init__(self, **kwargs):
        super(DummyProblem, self).__init__(**kwargs)
        self.call_count = 0

    def get_initial_state(self) -> GraphSearchState:
        return DummyState(0, is_terminal=False)

    @collect_metrics
    def get_reward(self, state: GraphSearchState) -> (float, {}):
        self.call_count += 1
        return 1, {}


def test_metrics():
    call_metrics.reset()
    start = DummyState(0, is_terminal=False)
    start.get_next_actions()
    assert call_metrics.execution_count['get_next_actions'] == 1
    assert call_metrics.execution_time['get_next_actions'] > 0.1
    start.get_next_actions()
    assert call_metrics.execution_count['get_next_actions'] == 2
    assert call_metrics.execution_time['get_next_actions'] > 0.2

    problem = DummyProblem(reward_class=RawRewardFactory())
    game = MCTS(problem)
    game.run(num_mcts_samples=5)
    assert call_metrics.execution_count['get_reward'] == problem.call_count
